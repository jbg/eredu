//! conditional Qwen semantic ingress retained across shared decoder spans.

use super::*;
use eredu_runtime::media_prefill::{MediaIngressError, PrefillIngressArchitecture};
use eredu_runtime::{PreparedInputInspector, PreparedModelInput, SharedPreparedInputCacheIdentity};

/// Exact ordinary prepared media input and ordinary decoder positions.
///
/// Native input/control admission is intentionally not supplied by this type.
pub struct MediaPrefillPlan<T> {
    prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
    admitted: Option<crate::media_plan::AdmittedCompositeInput<QwenHybridInputPartPlan>>,
    original: Option<crate::media_plan::BoundPreparedMediaSemantics>,
    workspace: Option<(
        eredu_runtime::input::OriginalEncoderTableProjection,
        eredu_runtime::input::OriginalPreparedWorkspaceSource,
        Option<eredu_runtime::working_memory::WorkingMemoryUnquotedLease>,
    )>,
    geometry: eredu_core::InferenceGeometry,
    fingerprint: String,
    identity: Option<SharedPreparedInputCacheIdentity>,
    // Plan/source fields and their allocations retire before planning custody.
    _workspace_funding: (
        Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
        Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
    ),
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}

impl<T: Tensor> MediaPrefillPlan<T> {
    /// Admits the actual ordered input and retains its exact architecture admission.
    pub fn prepare(
        args: &ParsedHybridConfig,
        prepared: PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        let admitted = crate::media_plan::admit_qwen_hybrid_input(args, &prepared, inspector)
            .map_err(Error::backend)?;
        Self::prepare_admitted(args, prepared, admitted, geometry)
    }

    /// Reuses the exact paired ordinary admission supplied by selected execution.
    pub fn prepare_admitted(
        args: &ParsedHybridConfig,
        prepared: PreparedModelInput<T>,
        admitted: crate::media_plan::AdmittedCompositeInput<QwenHybridInputPartPlan>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        geometry.validate().map_err(Error::backend)?;
        PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::backend)?;
        if admitted.decoder_shape() != [geometry.batch_size, geometry.input_positions]
            || geometry.batch_size != 1
            || geometry.cached_positions != 0
        {
            return Err(Error::backend(
                "conditional Qwen media source geometry differs from its admission",
            ));
        }
        // Ordinary text keeps the existing token source; raw media must have a
        // real active encoder. Projected parts inside this media request retain
        // their existing zero-token/ordinary-position semantics.
        if !admitted
            .parts()
            .iter()
            .any(|part| matches!(part, QwenHybridInputPartPlan::Media { .. }))
        {
            return Err(Error::backend(
                "conditional Qwen retained media source requires admitted raw media",
            ));
        }
        Ok(Self {
            prepared: prepared.into(),
            admitted: Some(admitted),
            original: None,
            workspace: None,
            geometry,
            fingerprint: super::super::config::conditional_prompt_cache_architecture_fingerprint(
                args,
            ),
            identity: None,
            _workspace_funding: (None, None),
            metadata: None,
        })
    }

    fn input(&self) -> Result<PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>, String> {
        self.input_with_diagnostic(str::to_owned)
    }

    fn input_with_metadata(
        &self,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>, Error> {
        metadata.controls::<PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>>()?;
        self.input_with_diagnostic(|message| metadata.error(format_args!("{message}")))
    }

    fn input_with_diagnostic<E>(
        &self,
        diagnostic: impl FnOnce(&'static str) -> E,
    ) -> Result<PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>, E> {
        match (&self.admitted, &self.original) {
            (Some(admitted), None) => {
                PreparedCompositeInput::new_with_diagnostic(&self.prepared, admitted, diagnostic)
            }
            (None, Some(original)) => PreparedCompositeInput::from_original_with_diagnostic(
                &self.prepared,
                original,
                diagnostic,
            ),
            _ => unreachable!("closed legacy/compiled semantic storage"),
        }
    }
    fn from_original(
        prepared: eredu_runtime::input::PreparedModelInputOwner<T>,
        original: crate::media_plan::BoundPreparedMediaSemantics,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, eredu_runtime::working_memory::OriginalCompositeSemanticStorageError> {
        if geometry.batch_size != 1
            || geometry.cached_positions != 0
            || geometry.input_positions != original.decoder_positions() as u64
            || prepared.len() != original.records().len()
            || geometry.validate_fixed().is_err()
        {
            return Err(
                original.reject(crate::media_plan::MediaSemanticError::input(
                    "compiled media source geometry mismatch",
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

    /// Retains the existing whole-input cache identity until final commit.
    pub fn with_shared_cache_identity(
        mut self,
        identity: SharedPreparedInputCacheIdentity,
    ) -> Result<Self, Error> {
        if identity.prepared() != self.prepared.identity() {
            return Err(Error::backend(
                "media cache identity differs from original prepared input",
            ));
        }
        self.identity = Some(identity);
        Ok(self)
    }
}

pub(super) struct PendingMedia<T> {
    input: PreparedPendingInput<T>,
}
impl<T> PendingMedia<T> {
    pub(super) fn values(&self) -> impl Iterator<Item = &T> {
        self.input
            .tokens
            .iter()
            .chain(self.input.projected.iter().flatten())
            .chain(self.input.pixels.iter())
    }
}

/// Compact projected media and exact original semantic parts, without decoder masks.
pub struct MediaIngress<T> {
    pending: PendingMedia<T>,
    vision: T,
    deepstack: Vec<T>,
}

impl<B, S> PrefillIngressArchitecture<B, S> for ConditionalLayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
    fn validate_ingress_plan(&self, plan: &Self::IngressPlan) -> Result<(), Error> {
        if plan.original.is_none()
            && plan.fingerprint
                != super::super::config::conditional_prompt_cache_architecture_fingerprint(
                    &self.parsed,
                )
        {
            return Err(Error::backend(
                "conditional Qwen media source belongs to another architecture",
            ));
        }
        plan.input().map_err(Error::backend)?;
        Ok(())
    }
    fn validate_ingress_plan_with_metadata(
        &self,
        plan: &Self::IngressPlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), Error> {
        if !context.uses_checked_metadata() {
            return <Self as PrefillIngressArchitecture<B, S>>::validate_ingress_plan(self, plan);
        }
        // The original binding already authenticates the selected architecture;
        // the ordinary branch's fingerprint is intentionally absent in this plan.
        let original = plan
            .original
            .as_ref()
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
        PreparedCompositeInput::<_, QwenHybridInputPartPlan>::from_original_with_diagnostic(
            &plan.prepared,
            original,
            |message| context.metadata_error(format_args!("{message}")),
        )?;
        Ok(())
    }
    fn ingress_error_with_metadata(
        error: MediaIngressError,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Error {
        context.metadata_source(error)
    }
    fn ingress_error(error: MediaIngressError) -> Error {
        Error::backend_source(error)
    }

    fn begin_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let metadata = crate::decoder::identity::Metadata::new(
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(
            LayeredForwardState<B::Tensor, Self::ForwardContext>,
            StateLayout,
        )>()?;
        let owned;
        let expected = match self.partition_state_layout.as_ref() {
            Some(layout) => layout,
            None if parallel.is_some() => self
                .parallel_geometry
                .as_ref()
                .ok_or_else(|| {
                    metadata.error(format_args!(
                        "conditional media source has no selected TP geometry"
                    ))
                })?
                .state_layout(),
            None => {
                owned = match metadata.context() {
                    Some(context) => {
                        super::super::state_layout_with_metadata(&self.parsed.text, context)?
                    }
                    None => {
                        super::super::state_layout(&self.parsed.text).map_err(Error::backend)?
                    }
                };
                &owned
            }
        };
        if state.layout() != expected {
            return Err(metadata.error(format_args!(
                "conditional Qwen media source has foreign local state geometry"
            )));
        }
        if expected.layer(0).is_some()
            && state
                .layer(0)
                .map_err(|cause| metadata.error(format_args!("{cause}")))?
                .position()
                != 0
        {
            return Err(metadata.error(format_args!(
                "conditional Qwen media input cannot append to a populated cache"
            )));
        }
        let mut input = prepare_pending_input(
            plan.input_with_metadata(metadata)?,
            context,
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        )?;
        let pixels = input
            .pixels
            .take()
            .ok_or_else(|| metadata.error(format_args!("media input has no pixels")))?;
        let (hidden, vision_state) = if let Some((tables, _, _)) = &plan.workspace {
            self.static_modules
                .vision
                .begin_with_projected_tables(&pixels, tables, context)?
        } else if plan.prepared.original_encoder_tables().is_some() {
            self.static_modules.vision.begin_with_original_tables(
                &pixels,
                &plan.prepared,
                context,
                metadata.context(),
            )?
        } else {
            let grids = plan
                .input()
                .map_err(Error::backend)?
                .qwen_parts()
                .flat_map(|part| part.grid.iter())
                .collect::<Vec<_>>();
            self.static_modules.vision.begin(
                VisionInput {
                    pixels: &pixels,
                    grid: &grids,
                },
                context,
            )?
        };
        Ok(LayeredForwardState {
            hidden: hidden.clone(),
            context: ConditionalForwardContext {
                metadata: plan.metadata.clone(),
                mask: None,
                tokens: None,
                parts: Vec::new(),
                embedded: None,
                mode: ForwardMode::Target,
                target_hidden: None,
                pending_media: Some(PendingMedia { input }),
                media_span: false,
                span_assembled: false,
                vision_state: Some(vision_state),
                vision_initial: Some(hidden),
                vision_output: None,
                deepstack: Vec::new(),
                visual_mask: None,
            },
        })
    }

    fn begin_ingress_received(
        &mut self,
        plan: &Self::IngressPlan,
        received: &B::Tensor,
        encoder_continuation: bool,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let metadata = crate::decoder::identity::Metadata::new(
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(
            LayeredForwardState<B::Tensor, Self::ForwardContext>,
            StateLayout,
        )>()?;
        let owned;
        let expected = match self.partition_state_layout.as_ref() {
            Some(layout) => layout,
            None if parallel.is_some() => self
                .parallel_geometry
                .as_ref()
                .ok_or_else(|| {
                    metadata.error(format_args!(
                        "conditional media source has no selected TP geometry"
                    ))
                })?
                .state_layout(),
            None => {
                owned = match metadata.context() {
                    Some(context) => {
                        super::super::state_layout_with_metadata(&self.parsed.text, context)?
                    }
                    None => {
                        super::super::state_layout(&self.parsed.text).map_err(Error::backend)?
                    }
                };
                &owned
            }
        };
        if state.layout() != expected {
            return Err(metadata.error(format_args!(
                "conditional Qwen media source has foreign local state geometry"
            )));
        }
        if expected.layer(0).is_some()
            && state
                .layer(0)
                .map_err(|cause| metadata.error(format_args!("{cause}")))?
                .position()
                != 0
        {
            return Err(metadata.error(format_args!(
                "conditional Qwen media input cannot append to a populated cache"
            )));
        }
        let mut input = prepare_pending_input(
            plan.input_with_metadata(metadata)?,
            context,
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        )?;
        // Tokens and compact metadata are prepared here; no patch/position projection.
        input.pixels = None;
        let vision_state = if encoder_continuation {
            if let Some(tables) = plan.prepared.original_encoder_tables() {
                Some(
                    self.static_modules
                        .vision
                        .continuation_state_with_original_tables(
                            &plan.prepared,
                            tables.layout().patches(),
                            context,
                            metadata.context(),
                        )?,
                )
            } else {
                let grids = plan
                    .input()
                    .map_err(Error::backend)?
                    .qwen_parts()
                    .flat_map(|part| part.grid.iter())
                    .collect::<Vec<_>>();
                let patches = grids
                    .iter()
                    .try_fold(0i32, |total, &(t, h, w)| {
                        t.checked_mul(h)
                            .and_then(|x| x.checked_mul(w))
                            .and_then(|x| total.checked_add(x))
                    })
                    .ok_or_else(|| Error::backend("media patch count overflows"))?;
                Some(
                    self.static_modules
                        .vision
                        .continuation_state(&grids, patches, context)?,
                )
            }
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden: received.clone(),
            context: ConditionalForwardContext {
                metadata: plan.metadata.clone(),
                mask: None,
                tokens: None,
                parts: Vec::new(),
                embedded: None,
                mode: ForwardMode::Target,
                target_hidden: None,
                pending_media: Some(PendingMedia { input }),
                media_span: false,
                span_assembled: false,
                vision_state,
                vision_initial: None,
                vision_output: None,
                deepstack: Vec::new(),
                visual_mask: None,
            },
        })
    }

    fn retain_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Ingress, Error> {
        let metadata = crate::decoder::identity::Metadata::new(
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<Self::Ingress>()?;
        // Validate all components before moving any ownership out of context.
        if forward.pending_media.is_none()
            || forward.vision_output.is_none()
            || forward.deepstack.len()
                != self
                    .parsed
                    .vision
                    .as_ref()
                    .expect("validated conditional vision")
                    .deepstack_layers()
                    .len()
        {
            return Err(metadata.error(format_args!("conditional Qwen media cut is incomplete")));
        }
        Ok(MediaIngress {
            pending: forward
                .pending_media
                .take()
                .expect("validated pending source"),
            vision: forward
                .vision_output
                .take()
                .expect("validated projected output"),
            deepstack: std::mem::take(&mut forward.deepstack),
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
        let metadata = crate::decoder::identity::Metadata::new(
            plan.metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(
            LayeredForwardState<B::Tensor, Self::ForwardContext>,
            PreparedPart<B::Tensor>,
            Vec<B::Tensor>,
            [Index; 3],
        )>()?;
        let start = i32::try_from(span.input.start).map_err(|cause| metadata.source(cause))?;
        let end = i32::try_from(span.input.end)
            .map_err(|cause| metadata.error(format_args!("{cause}")))?;
        if start < 0 || end <= start || end as u64 > plan.geometry.input_positions {
            return Err(metadata.error(format_args!("invalid conditional Qwen decoder span")));
        }
        let capacity = ingress.pending.input.kinds.len();
        let media_capacity = ingress
            .pending
            .input
            .kinds
            .iter()
            .filter(|kind| {
                matches!(
                    kind,
                    PreparedInputKind::Image(..) | PreparedInputKind::Video(..)
                )
            })
            .count();
        let mut parts = metadata.vector(capacity)?;
        let mut media_values = metadata.vector(media_capacity)?;
        let mut media_ranges = metadata.vector(media_capacity)?;
        let mut decoder_offset = 0;
        let mut media_offset = 0;
        for kind in &ingress.pending.input.kinds {
            let token = match *kind {
                PreparedInputKind::Text(i)
                | PreparedInputKind::Projected(i, _)
                | PreparedInputKind::Image(i, _)
                | PreparedInputKind::Video(i, _) => i,
            };
            let original = &ingress.pending.input.tokens[token];
            let length = original.dim(1);
            let local_start = (start - decoder_offset).max(0).min(length);
            let local_end = (end - decoder_offset).max(0).min(length);
            let is_media = matches!(
                kind,
                PreparedInputKind::Image(..) | PreparedInputKind::Video(..)
            );
            if local_start < local_end {
                let tokens = original.index(
                    &[Index::Full, Index::Range(local_start, local_end)],
                    context,
                )?;
                match *kind {
                    PreparedInputKind::Text(_) => {
                        let embeddings = match parallel {
                            Some(parallel) => B::vocabulary_parallel_lookup(
                                &mut self.static_modules.text.embeddings,
                                &tokens,
                                EmbeddingLookupPolicy::Strict,
                                parallel,
                                context,
                            )?,
                            None => self
                                .static_modules
                                .text
                                .embeddings
                                .forward(&tokens, context)?,
                        };
                        parts.push(PreparedPart::Text { tokens, embeddings });
                    }
                    PreparedInputKind::Projected(_, original) => {
                        let embeddings = ingress.pending.input.projected[original]
                            .as_ref()
                            .expect("prepared projected input owns its value")
                            .index(
                                &[
                                    Index::Full,
                                    Index::Range(local_start, local_end),
                                    Index::Full,
                                ],
                                context,
                            )?;
                        parts.push(PreparedPart::Text { tokens, embeddings });
                    }
                    PreparedInputKind::Image(..) | PreparedInputKind::Video(..) => {
                        let range = (media_offset + local_start, media_offset + local_end);
                        media_values.push(ingress.vision.index(
                            &[Index::Full, Index::Range(range.0, range.1), Index::Full],
                            context,
                        )?);
                        media_ranges.push(range);
                        parts.push(PreparedPart::Media { tokens });
                    }
                }
            }
            decoder_offset += length;
            if is_media {
                media_offset += length;
            }
        }
        let vision_output = join_media(media_values, context)?;
        let mut deepstack = metadata.vector(ingress.deepstack.len())?;
        if !media_ranges.is_empty() {
            for features in &ingress.deepstack {
                let mut values = metadata.vector(media_ranges.len())?;
                for &(start, end) in &media_ranges {
                    values.push(features.index(
                        &[Index::Full, Index::Range(start, end), Index::Full],
                        context,
                    )?);
                }
                deepstack.push(join_media(values, context)?.expect("nonempty media ranges"));
            }
        }
        let initial = match parts
            .first()
            .ok_or_else(|| metadata.error(format_args!("empty media decoder span")))?
        {
            PreparedPart::Text { embeddings, .. } => embeddings.clone(),
            PreparedPart::Media { .. } => vision_output
                .as_ref()
                .expect("media span has projection")
                .clone(),
        };
        let mut forward = ConditionalForwardContext {
            metadata: plan.metadata.clone(),
            mask: if end - start > 1 {
                Some(B::causal_mask(
                    end - start,
                    i32::try_from(span.position)
                        .map_err(|cause| metadata.error(format_args!("{cause}")))?,
                    None,
                    context,
                )?)
            } else {
                None
            },
            tokens: None,
            parts,
            embedded: None,
            mode: ForwardMode::Target,
            target_hidden: None,
            pending_media: None,
            media_span: true,
            span_assembled: false,
            vision_state: None,
            vision_initial: None,
            vision_output,
            deepstack,
            visual_mask: None,
        };
        // Reuse the exact ordinary ordered assembly equation once. The selected
        // group traversal still owns the group observation and all decoder units.
        let hidden = <Self as LayeredArchitecture<B, S>>::begin_execution_group(
            self,
            1,
            &initial,
            &[],
            state,
            &mut forward,
            context,
        )?;
        if media_ranges.is_empty() {
            // Reuse the already reserved output vector; no media values were
            // inserted on this branch. Build each zero only after its slot exists.
            for _ in 0..ingress.deepstack.len() {
                forward.deepstack.push(hidden.zeros_like(context)?);
            }
        }
        forward.span_assembled = true;
        Ok(LayeredForwardState {
            hidden,
            context: forward,
        })
    }

    fn visit_ingress_roots(ingress: &Self::Ingress, visitor: &mut dyn FnMut(&B::Tensor)) {
        visitor(&ingress.vision);
        for value in &ingress.deepstack {
            visitor(value);
        }
        for value in ingress.pending.values() {
            visitor(value);
        }
    }
}

fn join_media<T: Tensor>(mut values: Vec<T>, context: &T::Context) -> Result<Option<T>, Error> {
    match values.len() {
        0 => Ok(None),
        1 => Ok(values.pop()),
        _ => T::concatenate(&values, 1, context).map(Some),
    }
}

impl<B> ConditionalLayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    pub(super) fn ingress_group_collective_waves(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, QwenHybridInputPartPlan>,
        tensor_partitions: usize,
        pipeline_stages: usize,
        include_text_ingress: bool,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        if group != 0 || tensor_partitions <= 1 || pipeline_stages <= 1 {
            return Ok(None);
        }
        let vision = self.parsed.vision.as_ref().ok_or_else(|| {
            "conditional Qwen collective schedule has no vision config".to_owned()
        })?;
        let mut patch_positions = 0_u64;
        let mut projected_positions = 0_u64;
        for part in input.qwen_parts() {
            if part.role != crate::media_plan::qwen::QwenPartRole::Encoded {
                continue;
            }
            projected_positions = projected_positions
                .checked_add(part.positions)
                .ok_or_else(|| "conditional Qwen projected positions overflowed".to_owned())?;
            for (time, height, width) in part.grid.iter() {
                let patches = u64::try_from(time)
                    .ok()
                    .and_then(|time| {
                        u64::try_from(height)
                            .ok()
                            .and_then(|height| time.checked_mul(height))
                    })
                    .and_then(|area| {
                        u64::try_from(width)
                            .ok()
                            .and_then(|width| area.checked_mul(width))
                    })
                    .ok_or_else(|| "conditional Qwen patch-grid geometry overflowed".to_owned())?;
                patch_positions = patch_positions
                    .checked_add(patches)
                    .ok_or_else(|| "conditional Qwen patch positions overflowed".to_owned())?;
            }
        }
        let patch_positions = i32::try_from(patch_positions)
            .map_err(|_| "conditional Qwen patch positions exceed i32".to_owned())?;
        let projected_positions = i32::try_from(projected_positions)
            .map_err(|_| "conditional Qwen projected positions exceed i32".to_owned())?;
        if patch_positions == 0 || projected_positions == 0 {
            return Ok(Some(vec![Vec::new(); pipeline_stages]));
        }
        let layers = vision.layer_count();
        if layers < pipeline_stages {
            return Err(
                "conditional Qwen vision collective schedule has fewer units than PP stages".into(),
            );
        }
        let patch_shape = vec![patch_positions, vision.hidden_size];
        let projected_shape = vec![projected_positions, vision.out_hidden_size];
        let mut ingress_operations = Vec::new();
        if include_text_ingress {
            for (part, plan) in input.prepared().parts().iter().zip(input.qwen_parts()) {
                if plan.role == crate::media_plan::qwen::QwenPartRole::Tokens {
                    let batch = part.payload().value().dim(0);
                    let positions = i32::try_from(plan.positions)
                        .map_err(|_| "conditional Qwen text positions exceed i32".to_owned())?;
                    ingress_operations.push(
                        crate::composite_execution::CompositeTensorCollective::Sum {
                            shape: vec![batch, positions, self.parsed.text.hidden_size],
                        },
                    );
                }
            }
        }
        let mut stages = Vec::with_capacity(pipeline_stages);
        for stage in 0..pipeline_stages {
            let range =
                eredu_core::balanced_contiguous_range(layers, pipeline_stages, stage, false)
                    .map_err(|error| error.to_string())?;
            let mut operations = ingress_operations.clone();
            for layer in range.clone() {
                operations.extend([
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: patch_shape.clone(),
                    },
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: patch_shape.clone(),
                    },
                ]);
                if vision
                    .layer_policy(layer)
                    .is_some_and(|policy| policy.deepstack_merger.is_some())
                {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: projected_shape.clone(),
                    });
                }
            }
            if range.end == layers {
                operations.push(crate::composite_execution::CompositeTensorCollective::Sum {
                    shape: projected_shape.clone(),
                });
            }
            stages.push(operations);
        }
        Ok(Some(stages))
    }
}

impl<B, S> crate::composite_execution::CompositeMediaIngressArchitecture<B, S>
    for ConditionalLayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
        original.bind_conditional(admission, blueprint, source, binding)
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
        let (mut plan, workspace, funding) = input.construct_plan(None, |prepared, original| {
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
            .construct_plan(Some(context), |prepared, original| {
                MediaPrefillPlan::from_original(prepared, original, geometry)
            })?;
        plan.workspace = Some(workspace);
        plan._workspace_funding = funding;
        Ok(plan)
    }
    fn prepared_ingress_input(
        plan: &Self::IngressPlan,
    ) -> PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan> {
        plan.input().expect("plan retains its exact admission")
    }
    fn media_group_collective_waves(
        &self,
        plan: &Self::IngressPlan,
        group: usize,
        tensor_partitions: usize,
        pipeline_stages: usize,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        self.ingress_group_collective_waves(
            group,
            <Self as crate::composite_execution::CompositeMediaIngressArchitecture<B, S>>::prepared_ingress_input(plan),
            tensor_partitions,
            pipeline_stages,
            false,
        )
    }
    fn media_primary_ingress_collectives(
        &self,
        plan: &Self::IngressPlan,
        span: &eredu_runtime::prefill::PrefillChunk,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        let mut offset = 0u64;
        let mut positions = Vec::new();
        for part in plan.input()?.qwen_parts() {
            let length = part.positions;
            let end = offset
                .checked_add(length)
                .ok_or("media decoder positions overflow")?;
            if part.role == crate::media_plan::qwen::QwenPartRole::Tokens {
                let start = offset.max(span.input.start);
                let stop = end.min(span.input.end);
                if start < stop {
                    positions.push(stop - start);
                }
            }
            offset = end;
        }
        crate::composite_execution::segmented_token_ingress_collectives(
            positions,
            self.parsed.text.hidden_size,
            tensor_partitions,
        )
    }
}
