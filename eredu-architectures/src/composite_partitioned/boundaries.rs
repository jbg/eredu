//! Retained architecture-authored schemas for composite pipeline edges.
use std::collections::BTreeMap;

use eredu_runtime::{
    BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype, BoundaryTensorSpec,
    BoundaryWireSchema, ExecutionGraph, ExecutionUnitLayout,
};

/// Sequence bound for an encoder continuation or learned-context edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeSequenceExtent {
    /// Uses the admitted decoder invocation length.
    Decoder,
    /// Uses an architecture-declared fixed maximum encoder length.
    Fixed(i32),
    /// Expands the admitted decoder length by an exact patch/merge factor.
    DecoderMultiple(i32),
}
impl CompositeSequenceExtent {
    fn resolve(self, decoder: i32) -> Result<i32, String> {
        let result = match self {
            Self::Decoder => Some(decoder),
            Self::Fixed(value) => Some(value),
            Self::DecoderMultiple(factor) => decoder.checked_mul(factor),
        };
        result
            .filter(|value| *value > 0)
            .ok_or_else(|| "composite boundary sequence extent is invalid or overflows i32".into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Schemas {
    Fixed(BoundaryWireSchema, CompositeSequenceExtent),
    AfterUnits(Vec<(BoundaryWireSchema, CompositeSequenceExtent)>),
}

/// Exact schema for one semantic edge, including source-unit-dependent context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositePartitionBoundary {
    schemas: Schemas,
}
impl CompositePartitionBoundary {
    /// Declares one unchanged schema for every owner of this edge.
    pub fn fixed(schema: BoundaryWireSchema, sequence: CompositeSequenceExtent) -> Self {
        Self {
            schemas: Schemas::Fixed(schema, sequence),
        }
    }
    /// Declares the context after each source unit, in architecture order.
    /// This retains changing encoder widths and accumulated learned context.
    pub fn after_units(
        schemas: Vec<BoundaryWireSchema>,
        sequence: CompositeSequenceExtent,
    ) -> Result<Self, String> {
        Self::after_unit_geometry(
            schemas
                .into_iter()
                .map(|schema| (schema, sequence))
                .collect(),
        )
    }

    /// Declares each source unit's schema and sequence bound together. Patch
    /// pyramids can change both frame rate and activation width at a cut.
    pub fn after_unit_geometry(
        geometry: Vec<(BoundaryWireSchema, CompositeSequenceExtent)>,
    ) -> Result<Self, String> {
        if geometry.is_empty() {
            return Err("composite boundary has no source-unit schemas".into());
        }
        Ok(Self {
            schemas: Schemas::AfterUnits(geometry),
        })
    }
}

/// Family-independent transport declarations keyed by source/destination group.
/// Decoder continuations retain their ordinary typed state boundary; these
/// declarations describe optional encoders and cross-group learned context.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompositePartitionBoundaries {
    edges: BTreeMap<(usize, usize), CompositePartitionBoundary>,
}
impl CompositePartitionBoundaries {
    /// Adds one exact semantic edge without assuming any canonical group order.
    pub fn with_boundary(
        mut self,
        source_group: usize,
        destination_group: usize,
        boundary: CompositePartitionBoundary,
    ) -> Result<Self, String> {
        match &boundary.schemas {
            Schemas::Fixed(_, sequence) => {
                sequence.resolve(1)?;
            }
            Schemas::AfterUnits(geometry) => {
                for (_, sequence) in geometry {
                    sequence.resolve(1)?;
                }
            }
        }
        if source_group != destination_group && matches!(boundary.schemas, Schemas::AfterUnits(_)) {
            return Err("source-unit schemas require an intra-group continuation".into());
        }
        if self
            .edges
            .insert((source_group, destination_group), boundary)
            .is_some()
        {
            return Err("composite boundary edge was declared twice".into());
        }
        Ok(self)
    }

    pub(crate) fn validate(
        &self,
        graph: &ExecutionGraph,
        units: &ExecutionUnitLayout,
    ) -> Result<(), String> {
        for (&(source, destination), boundary) in &self.edges {
            if source >= graph.groups().len()
                || destination >= graph.groups().len()
                || (source != destination
                    && !graph
                        .dependencies(destination)
                        .is_some_and(|values| values.contains(&source)))
            {
                return Err("composite boundary does not name an execution-graph edge".into());
            }
            let count = units
                .group_range(source)
                .ok_or("composite source has no unit range")?
                .len();
            if count == 0
                || matches!(&boundary.schemas, Schemas::AfterUnits(schemas) if schemas.len() != count)
            {
                return Err("composite boundary schemas differ from source-unit geometry".into());
            }
        }
        Ok(())
    }

    /// Resolves an admitted pipeline cut using only retained schemas and geometry.
    pub fn resolve(
        &self,
        source_group: usize,
        destination_group: usize,
        source_pipeline: usize,
        pipeline_stages: usize,
        maximum_decoder_sequence: i32,
    ) -> Result<Option<(BoundaryWireSchema, i32)>, String> {
        let Some(boundary) = self.edges.get(&(source_group, destination_group)) else {
            return Ok(None);
        };
        let (schema, sequence) = match &boundary.schemas {
            Schemas::Fixed(schema, sequence) => (schema, sequence),
            Schemas::AfterUnits(schemas) => {
                let range = eredu_core::balanced_contiguous_range(
                    schemas.len(),
                    pipeline_stages,
                    source_pipeline,
                    false,
                )
                .map_err(|error| error.to_string())?;
                let (schema, sequence) = &schemas[range.end - 1];
                (schema, sequence)
            }
        };
        Ok(Some((
            schema.clone(),
            sequence.resolve(maximum_decoder_sequence)?,
        )))
    }
}

fn hidden(width: i32, unbatched: bool) -> Result<BoundaryWireSchema, String> {
    let mut shape = Vec::new();
    if !unbatched {
        shape.push(Dim::Batch);
    }
    shape.extend([Dim::Sequence, Dim::Fixed(width)]);
    BoundaryWireSchema::new(
        "composite-group-activation-v1",
        BoundaryTensorSpec::new("hidden", shape, Dtype::Activation),
        [],
    )
    .map_err(|error| error.to_string())
}

/// Existing family projections author transport once during header preparation.
/// Later rank admission consumes the retained declarations, not configuration.
pub(crate) fn from_config(
    config: &crate::replicated_text::CompositeConfig<'_>,
) -> Result<CompositePartitionBoundaries, String> {
    use crate::replicated_text::CompositeConfig;
    use CompositeSequenceExtent::{Decoder, DecoderMultiple, Fixed};
    let mut plan = CompositePartitionBoundaries::default();
    match config {
        CompositeConfig::NemotronLabsDiffusion(args) => {
            let factor = args
                .spatial_merge_size
                .checked_mul(args.spatial_merge_size)
                .ok_or("image patch expansion overflows i32")?;
            let factor = i32::try_from(factor).map_err(|_| "image patch expansion exceeds i32")?;
            for (source, destination, extent) in [
                (0, 0, DecoderMultiple(factor)),
                (0, 1, DecoderMultiple(factor)),
                (1, 2, Decoder),
            ] {
                plan = plan.with_boundary(
                    source,
                    destination,
                    CompositePartitionBoundary::fixed(
                        crate::nemotron_labs_diffusion::composite::group_boundary_schema(
                            args,
                            source,
                            destination,
                        )
                        .map_err(|error| error.to_string())?,
                        extent,
                    ),
                )?;
            }
        }
        CompositeConfig::Diffusion(args) => {
            if let Some(vision) = &args.vision {
                let factor = vision
                    .pooling_kernel_size
                    .checked_mul(vision.pooling_kernel_size)
                    .ok_or("image patch expansion overflows")?;
                plan = plan.with_boundary(
                    0,
                    0,
                    CompositePartitionBoundary::fixed(
                        hidden(vision.hidden_size, false)?,
                        DecoderMultiple(factor),
                    ),
                )?;
                plan = plan.with_boundary(
                    0,
                    1,
                    CompositePartitionBoundary::fixed(
                        hidden(args.text.hidden_size, false)?,
                        Decoder,
                    ),
                )?;
            }
        }
        CompositeConfig::Gemma4(args) => {
            for (group, width) in [
                (0, args.vision.as_ref().map(|vision| vision.hidden_size)),
                (1, args.audio.as_ref().map(|audio| audio.hidden_size)),
            ] {
                if let Some(width) = width {
                    plan = plan.with_boundary(
                        group,
                        group,
                        CompositePartitionBoundary::fixed(hidden(width, false)?, Decoder),
                    )?;
                }
            }
        }
        CompositeConfig::Muse(args) => {
            if let Some(audio) = &args.audio_config {
                let mut factor = audio.samples_per_token().map_err(|error| error.to_string())?;
                let mut width = 1;
                let mut geometry = Vec::new();
                let schema = |width, continuation| BoundaryWireSchema::new(
                    if continuation {"muse.audio_continuation"} else {"muse.audio_to_decoder"},
                    BoundaryTensorSpec::new("hidden", [Dim::Batch, Dim::Sequence, Dim::Fixed(width)], Dtype::Activation), [],
                ).map_err(|error| error.to_string());
                for kind in audio.execution_units().map_err(|error| error.to_string())? {
                    match kind {
                        crate::muse::AudioUnitKind::Patch {stage} => {
                            factor /= audio.patch_sizes[stage]; width = audio.hidden_sizes[stage];
                        },
                        crate::muse::AudioUnitKind::Projection => width = audio.out_hidden_size,
                        crate::muse::AudioUnitKind::Convolution {..} => {},
                    }
                    geometry.push((schema(width, true)?, DecoderMultiple(factor)));
                }
                plan = plan.with_boundary(2, 2, CompositePartitionBoundary::after_unit_geometry(geometry)?)?
                    .with_boundary(2, 1, CompositePartitionBoundary::fixed(schema(args.hidden_size, false)?, Decoder))?;
            }
            if let Some(vision) = &args.vision_config {
                let factor = vision
                    .merge_size
                    .checked_mul(vision.merge_size)
                    .ok_or("Muse patch continuation factor overflows i32")?;
                for (destination, continuation, sequence) in
                    [(0, true, DecoderMultiple(factor)), (1, false, Decoder)]
                {
                    let schema =
                        crate::muse::model::vision_partition_boundary_schema(args, continuation)
                            .map_err(|error| error.to_string())?;
                    plan = plan.with_boundary(
                        0,
                        destination,
                        CompositePartitionBoundary::fixed(schema, sequence),
                    )?;
                }
            }
        }
        CompositeConfig::QwenVl(args) => {
            let schemas = (0..args.vision.layer_count())
                .map(|last| {
                    let count = (0..=last)
                        .filter(|index| {
                            args.vision
                                .layer_policy(*index)
                                .is_some_and(|policy| policy.deepstack_merger.is_some())
                        })
                        .count();
                    crate::qwen::vl::vision_partition_boundary_schema(args, true, count)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            plan = plan.with_boundary(
                0,
                0,
                CompositePartitionBoundary::after_units(schemas, Decoder)?,
            )?;
            let output = crate::qwen::vl::vision_partition_boundary_schema(
                args,
                false,
                args.vision.deepstack_layer_count(),
            )
            .map_err(|error| error.to_string())?;
            plan = plan.with_boundary(0, 1, CompositePartitionBoundary::fixed(output, Decoder))?;
        }
        CompositeConfig::QwenHybrid(args) => {
            if let Some(vision) = &args.vision {
                let schema = |continuation: bool, count: usize| {
                    let primary = if continuation {
                        vec![Dim::Sequence, Dim::Fixed(vision.hidden_size)]
                    } else {
                        vec![Dim::Batch, Dim::Sequence, Dim::Fixed(args.text.hidden_size)]
                    };
                    BoundaryWireSchema::new(
                        if continuation {
                            "qwen_conditional.vision_continuation"
                        } else {
                            "qwen_conditional.vision_to_decoder"
                        },
                        BoundaryTensorSpec::new("hidden", primary, Dtype::Activation),
                        (0..count).map(|index| {
                            BoundaryTensorSpec::new(
                                format!("deepstack.{index}"),
                                [Dim::Batch, Dim::Sequence, Dim::Fixed(args.text.hidden_size)],
                                Dtype::Activation,
                            )
                        }),
                    )
                    .map_err(|error| error.to_string())
                };
                let schemas = (0..vision.layer_count())
                    .map(|last| {
                        let count = (0..=last)
                            .filter(|index| {
                                vision
                                    .layer_policy(*index)
                                    .is_some_and(|policy| policy.deepstack_merger.is_some())
                            })
                            .count();
                        schema(true, count)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                plan = plan.with_boundary(
                    0,
                    0,
                    CompositePartitionBoundary::after_units(
                        schemas,
                        Fixed(vision.num_position_embeddings),
                    )?,
                )?;
                plan = plan.with_boundary(
                    0,
                    1,
                    CompositePartitionBoundary::fixed(
                        schema(false, vision.deepstack_layer_count())?,
                        Decoder,
                    ),
                )?;
            }
        }
        CompositeConfig::Inkling(args) => {
            if let Some(vision) = &args.vision_config {
                let schemas = vision
                    .layer_specs()
                    .iter()
                    .map(|spec| hidden(spec.1, false))
                    .collect::<Result<Vec<_>, _>>()?;
                plan = plan.with_boundary(
                    0,
                    0,
                    CompositePartitionBoundary::after_units(schemas, Decoder)?,
                )?;
            }
            // Audio embeddings are static modules: this execution group has
            // no units and therefore no inter-stage continuation.
        }
    }
    Ok(plan)
}
