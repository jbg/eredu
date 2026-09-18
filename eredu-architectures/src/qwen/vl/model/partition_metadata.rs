//! The same partition schemas and values with ordinary or funded destinations.
use super::*;
use crate::composite_execution::graph::Destination;
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    pub(super) fn unit_path_in(
        &self,
        group: usize,
        index: usize,
        destination: Destination<'_>,
    ) -> Result<String, Error> {
        destination.controls::<(&Self, usize, usize, String)>()?;
        let count = match group {
            0 => self.args.vision.layer_count(),
            1 => usize::try_from(self.args.text.num_hidden_layers)
                .map_err(|cause| destination.error(format_args!("{cause}")))?,
            _ => return Err(destination.error(format_args!("Qwen3-VL has two execution groups"))),
        };
        if index >= count {
            return Err(destination.error(format_args!("Qwen3-VL unit is outside its group")));
        }
        match group {
            0 => destination.text(format_args!("model.visual.blocks.{index}")),
            1 => destination.text(format_args!(
                "{}.layers.{index}",
                self.args.text.parameter_root
            )),
            _ => unreachable!(),
        }
    }
    pub(super) fn group_path_in(
        active: bool,
        path: &str,
        destination: Destination<'_>,
    ) -> Result<Option<String>, Error> {
        destination.controls::<(bool, &str, Option<String>)>()?;
        active
            .then(|| destination.text(format_args!("{path}")))
            .transpose()
    }
    pub(super) fn partition_boundary_schema_in(
        &self,
        source_group: usize,
        destination_group: usize,
        selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        batch: i32,
        source_sequence: i32,
        group_sequences: &[i32],
        continuation: Option<(i32, i32)>,
        destination: Destination<'_>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Error> {
        destination.controls::<(
            &Self,
            usize,
            i32,
            &[i32],
            Option<(i32, i32)>,
            PipelineBoundarySchema,
            eredu_runtime::BoundaryWireSchema,
            Vec<i32>,
            Option<eredu_runtime::ResolvedBoundaryWireSchema>,
        )>()?;
        if source_group == 1 && destination_group == 1 {
            let _media_sequence = *group_sequences.first().ok_or_else(|| {
                destination.error(format_args!(
                    "Qwen3-VL decoder continuation has no vision sequence authority"
                ))
            })?;
            let boundary = PipelineBoundarySchema::from_args(&self.args);
            let schema = match destination.0 {
                Some(context) => eredu_runtime::ArchitectureBoundary::wire_schema_with_metadata(
                    &boundary, context,
                )?,
                None => eredu_runtime::ArchitectureBoundary::wire_schema(&boundary)
                    .map_err(Error::backend_retained_source)?,
            };
            let auxiliary = selected.auxiliary().len();
            if auxiliary < 3 {
                return Err(destination.error(format_args!(
                    "Qwen3-VL decoder continuation omitted mRoPE boundary roles"
                )));
            }
            return destination.resolve_boundary(
                &schema,
                batch,
                &destination.collect(
                    std::iter::once(source_sequence)
                        .chain(std::iter::repeat_n(source_sequence, 3))
                        .chain(std::iter::repeat_n(source_sequence, auxiliary - 3)),
                )?,
            )
            .map(Some);
        }
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let same_group = source_group == destination_group;
        let schema = vision_schema(
            &self.args,
            same_group,
            selected.auxiliary().len(),
            destination,
        )?;
        let primary_sequence = if same_group {
            continuation
                .ok_or_else(|| {
                    destination.error(format_args!("Qwen3-VL continuation has no patch geometry"))
                })?
                .0
        } else {
            source_sequence
        };
        destination.resolve_boundary(
            &schema,
            batch,
            &destination.collect(std::iter::once(primary_sequence).chain(std::iter::repeat_n(
                source_sequence,
                selected.auxiliary().len(),
            )))?,
        )
        .map(Some)
    }

    pub(super) fn partition_boundary_values_in(
        &self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        hidden: &B::Tensor,
        forward: &ForwardContext<B::Tensor>,
        destination: Destination<'_>,
    ) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>, Error> {
        destination.controls::<(
            &Self,
            &ForwardContext<B::Tensor>,
            Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>,
            Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>,
            std::slice::Iter<'_, B::Tensor>,
        )>()?;
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let deepstack = if source_group == destination_group {
            forward
                .vision_state
                .as_ref()
                .ok_or_else(|| {
                    destination.error(format_args!("Qwen3-VL continuation has no vision state"))
                })?
                .deepstack_features()
        } else {
            &forward.deepstack
        };
        if deepstack.len() != schema.auxiliary().len() {
            return Err(destination.error(format_args!(
                "Qwen3-VL vision boundary has incomplete DeepStack context"
            )));
        }
        let mut values = destination.vector(
            1usize
                .checked_add(deepstack.len())
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        values.push(destination.boundary_value(schema.primary().role(), hidden.clone())?);
        for (spec, tensor) in schema.auxiliary().iter().zip(deepstack) {
            values.push(destination.boundary_value(spec.role(), tensor.clone())?);
        }
        Ok(Some(values))
    }
}

fn vision_schema(
    args: &ModelArgs,
    continuation: bool,
    deepstack_count: usize,
    destination: Destination<'_>,
) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
    // Fund the actual borrowed declaration before constructing its shape array.
    destination.controls::<(
        FamilySchema<'_>,
        [eredu_runtime::BoundaryTensorDimension; 3],
    )>()?;
    with_vision_schema(
        args,
        continuation,
        deepstack_count,
        |schema| match destination.0 {
            Some(context) => schema.with_metadata(context),
            None => schema.ordinary().map_err(Error::backend_retained_source),
        },
    )
}

/// One immutable family declaration consumed by both storage destinations.
struct FamilySchema<'a> {
    identity: &'static str,
    hidden: i32,
    fixed: &'a [crate::boundary_metadata::Field<'a>],
    repeated: Option<crate::boundary_metadata::Repeated<'a>>,
}
impl FamilySchema<'_> {
    fn primary(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden)
    }
    fn auxiliary(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        let mut fields = Vec::with_capacity(
            self.fixed.len() + self.repeated.map_or(0, |(_, count, _, _)| count),
        );
        for &(role, shape, dtype) in self.fixed {
            fields.push(eredu_runtime::BoundaryTensorSpec::new(
                role,
                shape.iter().copied(),
                dtype,
            ));
        }
        if let Some((prefix, count, shape, dtype)) = self.repeated {
            fields.extend((0..count).map(|index| {
                eredu_runtime::BoundaryTensorSpec::new(
                    format!("{prefix}.{index}"),
                    shape.iter().copied(),
                    dtype,
                )
            }));
        }
        fields
    }
    fn ordinary(
        &self,
    ) -> Result<eredu_runtime::BoundaryWireSchema, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::BoundaryWireSchema::new(self.identity, self.primary(), self.auxiliary())
    }
    fn with_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        crate::boundary_metadata::schema(
            context,
            self.identity,
            self.hidden,
            self.fixed,
            self.repeated,
        )
    }
}

pub(super) fn ordinary_vision_schema(
    args: &ModelArgs,
    continuation: bool,
    count: usize,
) -> Result<eredu_runtime::BoundaryWireSchema, eredu_runtime::ArchitectureBoundaryError> {
    with_vision_schema(args, continuation, count, |schema| schema.ordinary())
}
fn with_vision_schema<R>(
    args: &ModelArgs,
    continuation: bool,
    count: usize,
    consume: impl FnOnce(FamilySchema<'_>) -> R,
) -> R {
    use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
    consume(FamilySchema {
        identity: if continuation {
            "qwen_vl.vision_continuation"
        } else {
            "qwen_vl.vision_to_decoder"
        },
        hidden: if continuation {
            args.vision.hidden_size
        } else {
            args.text.hidden_size
        },
        fixed: &[],
        repeated: Some((
            "deepstack",
            count,
            &[Dim::Batch, Dim::Sequence, Dim::Fixed(args.text.hidden_size)],
            Dtype::Activation,
        )),
    })
}
impl PipelineBoundarySchema {
    fn with_family_schema<R>(&self, consume: impl FnOnce(FamilySchema<'_>) -> R) -> R {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        consume(FamilySchema {
            identity: "qwen_vl.decoder",
            hidden: self.hidden_size,
            fixed: &[
                (
                    "cosine",
                    &[Dim::Sequence, Dim::Fixed(self.head_dim)],
                    Dtype::Activation,
                ),
                (
                    "sine",
                    &[Dim::Sequence, Dim::Fixed(self.head_dim)],
                    Dtype::Activation,
                ),
                ("position_delta", &[Dim::Fixed(1)], Dtype::Int32),
            ],
            repeated: Some((
                "deepstack",
                self.deepstack_count,
                &[Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                Dtype::Activation,
            )),
        })
    }
    pub(super) fn primary_declaration(&self) -> eredu_runtime::BoundaryTensorSpec {
        self.with_family_schema(|schema| schema.primary())
    }
    pub(super) fn auxiliary_declarations(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        self.with_family_schema(|schema| schema.auxiliary())
    }
    pub(super) fn wire_declaration_with_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        // Three fixed shape arrays and one repeated shape are borrowed by the
        // common declaration; their exact frame is funded before construction.
        use eredu_runtime::BoundaryTensorDimension as Dim;
        Destination(Some(context)).controls::<(
            FamilySchema<'_>,
            [crate::boundary_metadata::Field<'_>; 3],
            [Dim; 2],
            [Dim; 2],
            [Dim; 1],
            [Dim; 3],
        )>()?;
        self.with_family_schema(|schema| schema.with_metadata(context))
    }
}
