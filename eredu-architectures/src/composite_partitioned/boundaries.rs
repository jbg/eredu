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
    Fixed(BoundaryWireSchema),
    AfterUnits(Vec<BoundaryWireSchema>),
}

/// Exact schema for one semantic edge, including source-unit-dependent context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositePartitionBoundary {
    schemas: Schemas,
    sequence: CompositeSequenceExtent,
}
impl CompositePartitionBoundary {
    /// Declares one unchanged schema for every owner of this edge.
    pub fn fixed(schema: BoundaryWireSchema, sequence: CompositeSequenceExtent) -> Self {
        Self {
            schemas: Schemas::Fixed(schema),
            sequence,
        }
    }
    /// Declares the context after each source unit, in architecture order.
    /// This retains changing encoder widths and accumulated learned context.
    pub fn after_units(
        schemas: Vec<BoundaryWireSchema>,
        sequence: CompositeSequenceExtent,
    ) -> Result<Self, String> {
        if schemas.is_empty() {
            return Err("composite boundary has no source-unit schemas".into());
        }
        Ok(Self {
            schemas: Schemas::AfterUnits(schemas),
            sequence,
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
        boundary.sequence.resolve(1)?;
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
        let schema = match &boundary.schemas {
            Schemas::Fixed(schema) => schema,
            Schemas::AfterUnits(schemas) => {
                let range = eredu_core::balanced_contiguous_range(
                    schemas.len(),
                    pipeline_stages,
                    source_pipeline,
                    false,
                )
                .map_err(|error| error.to_string())?;
                &schemas[range.end - 1]
            }
        };
        Ok(Some((
            schema.clone(),
            boundary.sequence.resolve(maximum_decoder_sequence)?,
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
            if let Some(vision) = &args.vision_config {
                let factor = vision
                    .merge_size
                    .checked_mul(vision.merge_size)
                    .ok_or("Muse patch continuation factor overflows i32")?;
                for (destination, continuation, sequence) in
                    [(0, true, DecoderMultiple(factor)), (1, false, Decoder)]
                {
                    let schema = crate::muse_glimmer::model::vision_partition_boundary_schema(
                        args,
                        continuation,
                    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::ExecutionGroupSpec;

    fn graph() -> (ExecutionGraph, ExecutionUnitLayout) {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::with_dependencies("target", ["vision"]),
                ExecutionGroupSpec::root("vision"),
            ],
            "target",
        )
        .unwrap();
        let units = ExecutionUnitLayout::new(&graph, [3, 5]).unwrap();
        (graph, units)
    }

    fn continuation(width: i32, side_outputs: usize) -> BoundaryWireSchema {
        BoundaryWireSchema::new(
            "vision.continuation",
            BoundaryTensorSpec::new(
                "hidden",
                [Dim::Sequence, Dim::Fixed(width)],
                Dtype::Activation,
            ),
            (0..side_outputs).map(|index| {
                BoundaryTensorSpec::new(
                    format!("context.{index}"),
                    [Dim::Batch, Dim::Sequence, Dim::Fixed(12)],
                    Dtype::Activation,
                )
            }),
        )
        .unwrap()
    }

    #[test]
    fn retained_edges_follow_group_identity_and_exact_uneven_pipeline_cuts() {
        let (graph, units) = graph();
        let schemas = vec![
            continuation(4, 0),
            continuation(4, 1),
            continuation(8, 1),
            continuation(8, 2),
            continuation(16, 2),
        ];
        let output = hidden(12, false).unwrap();
        let boundaries = CompositePartitionBoundaries::default()
            .with_boundary(
                1,
                1,
                CompositePartitionBoundary::after_units(
                    schemas.clone(),
                    CompositeSequenceExtent::Fixed(64),
                )
                .unwrap(),
            )
            .unwrap()
            .with_boundary(
                1,
                0,
                CompositePartitionBoundary::fixed(output.clone(), CompositeSequenceExtent::Decoder),
            )
            .unwrap();
        boundaries.validate(&graph, &units).unwrap();
        // Five units split over three owners as 2, 2, 1. The selected schema
        // includes precisely the side outputs produced before each cut.
        for (rank, last) in [1, 3, 4].into_iter().enumerate() {
            let (schema, extent) = boundaries.resolve(1, 1, rank, 3, 7).unwrap().unwrap();
            assert_eq!(schema, schemas[last]);
            assert_eq!(extent, 64);
            let resolved = schema.resolve(2, extent).unwrap();
            assert_eq!(resolved.primary().shape(), [64, [4, 8, 16][rank]]);
            assert_eq!(resolved.auxiliary().len(), [1, 2, 2][rank]);
        }
        assert_eq!(
            boundaries.resolve(1, 0, 2, 3, 7).unwrap(),
            Some((output, 7))
        );
        assert!(boundaries.resolve(0, 0, 0, 3, 7).unwrap().is_none());
        assert!(boundaries.resolve(1, 1, 0, 6, 7).is_err());
    }

    #[test]
    fn sequence_expansion_is_checked_before_boundary_resolution() {
        let boundaries = CompositePartitionBoundaries::default()
            .with_boundary(
                1,
                1,
                CompositePartitionBoundary::fixed(
                    hidden(8, true).unwrap(),
                    CompositeSequenceExtent::DecoderMultiple(4),
                ),
            )
            .unwrap();
        let (schema, extent) = boundaries.resolve(1, 1, 0, 2, 9).unwrap().unwrap();
        assert_eq!(
            schema.resolve(1, extent).unwrap().primary().shape(),
            [36, 8]
        );
        assert!(boundaries.resolve(1, 1, 0, 2, i32::MAX).is_err());
        assert!(boundaries.resolve(1, 1, 0, 2, 0).is_err());
        assert!(CompositePartitionBoundaries::default()
            .with_boundary(
                1,
                1,
                CompositePartitionBoundary::fixed(
                    hidden(8, true).unwrap(),
                    CompositeSequenceExtent::Fixed(0),
                ),
            )
            .is_err());
    }

    #[test]
    fn malformed_edges_and_source_geometry_are_rejected() {
        let (graph, units) = graph();
        let fixed = CompositePartitionBoundary::fixed(
            hidden(8, true).unwrap(),
            CompositeSequenceExtent::Decoder,
        );
        let reversed = CompositePartitionBoundaries::default()
            .with_boundary(0, 1, fixed.clone())
            .unwrap();
        assert!(reversed.validate(&graph, &units).is_err());
        assert!(reversed.with_boundary(0, 1, fixed).is_err());
        let truncated = CompositePartitionBoundary::after_units(
            vec![continuation(4, 0)],
            CompositeSequenceExtent::Decoder,
        )
        .unwrap();
        assert!(CompositePartitionBoundaries::default()
            .with_boundary(1, 0, truncated.clone())
            .is_err());
        assert!(CompositePartitionBoundaries::default()
            .with_boundary(1, 1, truncated)
            .unwrap()
            .validate(&graph, &units)
            .is_err());
    }

    #[test]
    fn inkling_retains_only_executable_vision_cuts_and_no_static_audio_continuation() {
        let args = crate::inkling::ModelArgs::from_hf_json(
            br#"{"text_config":{"hidden_size":16,"num_hidden_layers":1,"vocab_size":32,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"d_rel":2,
            "intermediate_size":24,"n_routed_experts":2,"num_experts_per_tok":1,
            "n_shared_experts":1},
            "audio_config":{"text_hidden_size":16,"num_codebooks":4,"codebook_size":8},
            "vision_config":{"text_hidden_size":16,"patch_size":40,"temporal_patch_size":2,
            "num_channels":3,"num_hidden_layers":4}}"#,
        )
        .unwrap();
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text", ["vision", "audio"]),
            ],
            "text",
        )
        .unwrap();
        let boundaries =
            from_config(&crate::replicated_text::CompositeConfig::Inkling(&args)).unwrap();
        let units = ExecutionUnitLayout::new(&graph, [4, 0, 1]).unwrap();
        boundaries.validate(&graph, &units).unwrap();
        assert!(boundaries.resolve(1, 1, 0, 2, 8).unwrap().is_none());
        let (schema, extent) = boundaries.resolve(0, 0, 0, 2, 8).unwrap().unwrap();
        assert_eq!(
            schema.resolve(1, extent).unwrap().primary().shape(),
            [1, 8, 512]
        );
    }
}
