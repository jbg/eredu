//! Prepared processor requests enter the same conditional graph and transactions.
use super::*;
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    media_plan::{AdmittedCompositeInput, QwenInputPartPlan},
    qwen4_exp::media::{MediaAdmissionConfig, MediaInputPartPlan},
};

impl<B, S> CompositeArchitecture<B, S> for ConditionalModel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    type InputPartPlan = MediaInputPartPlan<B::Tensor>;
    type AdmissionConfig = MediaAdmissionConfig;
    fn supports_retained_prefill() -> bool {
        true
    }

    type PrefillRequest = super::prefill::MediaPrefillRequest<B::Tensor>;

    fn prefill_chunk_limit(
        admitted: &AdmittedCompositeInput<Self::InputPartPlan>,
    ) -> Result<Option<usize>, Error> {
        MediaInputPartPlan::request(admitted.parts())
            .map(|request| Some(request.maximum_chunk_tokens()))
            .map_err(Error::backend_source)
    }

    fn prepare_prefill_request(
        prepared: PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Error> {
        PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::backend)?;
        super::prefill::MediaPrefillRequest::new(admitted, context).map_err(Error::backend_source)
    }

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.ingress.admission_config().clone()
    }
    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &PreparedModelInput<B::Tensor>,
        inspector: &impl PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<Self::InputPartPlan>, eredu_core::CapabilityError> {
        config
            .admit(input, inspector)
            .map(|proof| proof.into_composite())
            .map_err(|error| eredu_core::CapabilityError::UnsupportedInput {
                architecture: "qwen4_exp".into(),
                reason: error.to_string(),
            })
    }
    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        MediaInputPartPlan::request(input.admitted().parts())
            .map_err(Error::backend_source)?
            .token_ids(context)
    }
    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        group == TARGET_INDEX
            || (group == VISION_INDEX
                && MediaInputPartPlan::chunk(input.admitted().parts())
                    .is_none_or(|chunk| chunk.range.start == 0 && chunk.projected.is_none())
                && input
                    .admitted()
                    .parts()
                    .iter()
                    .any(|part| matches!(part.shared(), QwenInputPartPlan::Media { .. })))
    }
    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group == VISION_INDEX {
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part.shared() {
                    QwenInputPartPlan::Media { shape, .. } => Some(shape.decoder_positions),
                    _ => None,
                })
                .try_fold(0u64, |sum, n| sum.checked_add(n))
                .ok_or("projected media extent overflow")?
        } else {
            MediaInputPartPlan::chunk(input.admitted().parts())
                .map(|chunk| (chunk.range.end - chunk.range.start) as u64)
                .unwrap_or_else(|| input.admitted().decoder_positions())
        };
        i32::try_from(positions).map_err(|_| "composite boundary exceeds i32".into())
    }
    fn prepared_group_continuation_geometry(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<(i32, i32)>, String> {
        if group != VISION_INDEX {
            return Ok(None);
        }
        let patches = input
            .admitted()
            .parts()
            .iter()
            .filter_map(|part| match part.shared() {
                QwenInputPartPlan::Media { ingress, .. } => Some(&ingress.patch_grid),
                _ => None,
            })
            .flatten()
            .try_fold(0_i64, |total, &(time, height, width)| {
                i64::from(time)
                    .checked_mul(i64::from(height))
                    .and_then(|area| area.checked_mul(i64::from(width)))
                    .and_then(|count| total.checked_add(count))
                    .ok_or("vision patch extent overflow")
            })?;
        Ok(Some((
            i32::try_from(patches).map_err(|_| "vision patch extent exceeds i32")?,
            self.vision.hidden_size,
        )))
    }
    fn prepared_group_continuation_batched(&self, group: usize) -> bool {
        group != VISION_INDEX
    }
    fn prepared_primary_ingress_collectives(
        &self,
        _: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        _: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        // Vocabulary, input expansion and lexical computation are replicated.
        Ok(Some(Vec::new()))
    }
    fn routed_tensor_reductions(
        &self,
        unit: usize,
        routed: bool,
    ) -> Result<crate::partitioned_execution::RoutedTensorReductions, Error> {
        self.modules.target.routed_tensor_reductions(unit, routed)
    }
    fn routed_tensor_vocabulary_sharded(&self) -> bool {
        false
    }
    fn prepared_group_collective_waves(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        pipeline_stages: usize,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        if group != VISION_INDEX || tensor_partitions <= 1 || pipeline_stages <= 1 {
            return Ok(None);
        }
        if !<Self as CompositeArchitecture<B, S>>::should_execute_prepared_group(self, group, input)
        {
            return Ok(Some(vec![Vec::new(); pipeline_stages]));
        }
        let (patches, width) =
            <Self as CompositeArchitecture<B, S>>::prepared_group_continuation_geometry(
                self, group, input,
            )?
            .ok_or("missing vision continuation")?;
        let projected = <Self as CompositeArchitecture<B, S>>::prepared_group_boundary_sequence(
            self, group, input,
        )?;
        let mut stages = Vec::with_capacity(pipeline_stages);
        for stage in 0..pipeline_stages {
            let range = eredu_core::balanced_contiguous_range(
                self.vision.layer_count(),
                pipeline_stages,
                stage,
                false,
            )
            .map_err(|error| error.to_string())?;
            let mut operations = vec![
                crate::composite_execution::CompositeTensorCollective::Sum {
                    shape: vec![patches, width]
                };
                range.len() * 2
            ];
            if range.end == self.vision.layer_count() {
                operations.push(crate::composite_execution::CompositeTensorCollective::Sum {
                    shape: vec![projected, self.vision.out_hidden_size],
                });
            }
            stages.push(operations);
        }
        Ok(Some(stages))
    }
    fn partition_boundary_schema(
        &self,
        source_group: usize,
        destination_group: usize,
        _: &ResolvedBoundaryWireSchema,
        batch: i32,
        source_sequence: i32,
        _: &[i32],
        continuation: Option<(i32, i32)>,
    ) -> Result<Option<ResolvedBoundaryWireSchema>, Error> {
        if source_group == TARGET_INDEX && destination_group == TARGET_INDEX {
            return self
                .modules
                .target
                .spec
                .boundary_schema()?
                .wire_schema()
                .and_then(|schema| schema.resolve(batch, source_sequence))
                .map(Some)
                .map_err(Error::backend);
        }
        if source_group != VISION_INDEX || !matches!(destination_group, TARGET_INDEX | VISION_INDEX)
        {
            return Ok(None);
        }
        use eredu_runtime::{BoundaryTensorDimension as D, BoundaryTensorDtype as K};
        let (name, primary, sequence) = if destination_group == VISION_INDEX {
            let (patches, width) = continuation
                .ok_or_else(|| Error::backend("missing vision continuation geometry"))?;
            (
                "qwen4_exp.vision_continuation",
                BoundaryTensorSpec::new("hidden", [D::Sequence, D::Fixed(width)], K::Activation),
                patches,
            )
        } else {
            (
                "qwen4_exp.vision_to_target",
                BoundaryTensorSpec::new(
                    "hidden",
                    [D::Batch, D::Sequence, D::Fixed(self.vision.out_hidden_size)],
                    K::Activation,
                ),
                source_sequence,
            )
        };
        BoundaryWireSchema::new(name, primary, [])
            .and_then(|schema| schema.resolve(batch, sequence))
            .map(Some)
            .map_err(Error::backend)
    }
    fn partition_boundary_values(
        &self,
        source_group: usize,
        destination_group: usize,
        schema: &ResolvedBoundaryWireSchema,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<ArchitectureBoundaryValue<B::Tensor>>>, Error> {
        let mut values =
            vec![
                ArchitectureBoundaryValue::new(schema.primary().role(), hidden.clone())
                    .map_err(Error::backend)?,
            ];
        if source_group == TARGET_INDEX && destination_group == TARGET_INDEX {
            let request = &forward
                .target
                .as_ref()
                .ok_or_else(|| Error::backend("missing target transport context"))?
                .request;
            values.extend(
                self.modules
                    .target
                    .spec
                    .boundary_schema()?
                    .encode(request.boundary())
                    .map_err(Error::backend)?,
            );
        } else if source_group != VISION_INDEX
            || !matches!(destination_group, TARGET_INDEX | VISION_INDEX)
        {
            return Ok(None);
        }
        Ok(Some(values))
    }
    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        _: &ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
    ) -> Result<Option<B::Tensor>, Error> {
        if source_group != VISION_INDEX
            && !(source_group == TARGET_INDEX && destination_group == TARGET_INDEX)
        {
            return Ok(None);
        }
        if !matches!(destination_group, TARGET_INDEX | VISION_INDEX) {
            return Ok(None);
        }
        let mut values = values.into_iter();
        let hidden = values
            .next()
            .ok_or_else(|| Error::backend("missing conditional boundary activation"))?;
        if source_group == TARGET_INDEX {
            // Validation needs local committed state and a context; the ordinary
            // partition entry performs it before any receiving decoder unit.
            forward.incoming_target = Some(
                self.modules
                    .target
                    .spec
                    .boundary_schema()?
                    .decode(values.collect())
                    .map_err(Error::backend)?,
            );
        } else {
            if values.next().is_some() {
                return Err(Error::backend(
                    "unexpected vision boundary auxiliary values",
                ));
            }
            if destination_group == TARGET_INDEX {
                forward.projected = Some(hidden.clone());
            }
        }
        Ok(Some(hidden))
    }
    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let request =
            MediaInputPartPlan::request(input.admitted().parts()).map_err(Error::backend_source)?;
        request
            .validate_policy(self.ingress.admission_config())
            .map_err(Error::backend_source)?;
        if let Some(chunk) = MediaInputPartPlan::chunk(input.admitted().parts()) {
            return self.begin_forward(
                ConditionalInput::MediaChunk {
                    prepared: &chunk.prepared,
                    range: chunk.range.clone(),
                    projected: chunk.projected.as_ref(),
                },
                state,
                context,
            );
        }
        if input.admitted().decoder_positions() > self.modules.target.spec.limits.qsa.tokens as u64
        {
            return Err(Error::backend_source(
                super::super::media::MediaInputError::Geometry(
                    "composite prompt exceeds the admitted target invocation",
                ),
            ));
        }
        // The state frontier is needed even for text-only composite continuations.
        // Reject before request tensors, embeddings or a vision invocation exist.
        let offset = state.layer(0).map_err(Error::backend)?.position();
        self.modules
            .target
            .spec
            .limits
            .validate_history(offset, input.admitted().decoder_positions() as i32)
            .map_err(Error::backend_source)?;
        let prepared = request.prepare(context).map_err(Error::backend_source)?;
        let modalities = prepared.admission().active_modalities();
        if modalities.image || modalities.video {
            return self.begin_forward(ConditionalInput::Media(&prepared), state, context);
        }
        if self
            .partition_state
            .as_ref()
            .is_some_and(|partition| partition.global_layer_offset() != 0)
        {
            // Only the input owner embeds text. The receiving owner restores
            // original IDs and the media-derived rotary delta from its boundary.
            let input =
                <Self as ReplicatedTextArchitecture<B, S>>::text_input(prepared.token_ids(), None);
            return self.begin_forward(input, state, context);
        }
        // Ordinary token/projected-text continuation derives rotary positions from
        // committed state, including a previous media prompt's signed delta.
        let assembled = prepared
            .assemble(
                0..prepared.token_ids().dim(1),
                None,
                |ids| {
                    self.modules
                        .target
                        .decoder
                        .static_modules_mut()
                        .embeddings
                        .forward(ids, context)
                },
                context,
            )
            .map_err(Error::backend_source)?;
        assembled.with_target_input(|mut input| {
            input.rotary = None;
            input.position_delta = None;
            self.begin_forward(ConditionalInput::Target(input), state, context)
        })
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        <Self as LayeredArchitecture<B, S>>::prediction_target_capture(forward)
    }
}
