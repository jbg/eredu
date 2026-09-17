//! Metadata destination for the shared static parameter declaration worker.

use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};

pub(crate) enum ParameterGroupError {
    Ordinary(ParallelPlanError),
    Metadata(Error),
}
impl ParameterGroupError {
    pub(crate) fn ordinary(self) -> ParallelPlanError {
        match self {
            Self::Ordinary(cause) => cause,
            Self::Metadata(_) => {
                unreachable!("ordinary static groups have no metadata destination")
            }
        }
    }
    pub(crate) fn into_neural(self) -> Error {
        match self {
            Self::Ordinary(cause) => Error::backend(cause),
            Self::Metadata(cause) => cause,
        }
    }
}
#[derive(Clone, Copy)]
enum ShapePolicy {
    Embedding,
    Replicated,
    Head,
}
impl ShapePolicy {
    fn placement(self, shape: &[usize]) -> Result<MemberSharding, &'static str> {
        match self {
            Self::Embedding | Self::Head if shape.is_empty() => Err(match self {
                Self::Embedding => "decoder embedding parameter is scalar",
                _ => "decoder language-model head parameter is scalar",
            }),
            Self::Embedding | Self::Head => Ok(MemberSharding::Balanced { axis: 0 }),
            Self::Replicated => Ok(MemberSharding::Replicated),
        }
    }
}
fn group<T: Tensor, M: eredu_nn::Parameterized<T>>(
    name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    module: &M,
    policy: ShapePolicy,
    context: Option<&WorkspaceContext>,
) -> Result<ParameterGroupSpec, ParameterGroupError> {
    match context {
        Some(context) => eredu_runtime::module_parameter_group_with_metadata::<T, M>(
            name,
            role,
            module,
            context,
            |_, shape| {
                policy
                    .placement(shape)
                    .map_err(|cause| context.metadata_error(format_args!("{cause}")))
            },
        )
        .map_err(ParameterGroupError::Metadata),
        None => module_parameter_group::<T, M>(name.to_string(), role, module, |_, shape| {
            policy
                .placement(shape)
                .map_err(|cause| ParallelPlanError::InvalidTensor(cause.into()))
        })
        .map_err(ParameterGroupError::Ordinary),
    }
}

pub(crate) fn static_groups<B: NeuralBackend>(
    embeddings: &B::Embedding,
    norm: &B::Normalization,
    head: Option<&B::Linear>,
    parameter_root: &str,
    context: Option<&WorkspaceContext>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    let count = 2 + usize::from(head.is_some());
    let mut groups = match context {
        Some(context) => {
            let parts = [
                size_of::<Vec<ParameterGroupSpec>>(),
                size_of::<ParameterGroupError>(),
                size_of::<ShapePolicy>(),
                size_of::<Option<&WorkspaceContext>>(),
                size_of::<Result<Vec<ParameterGroupSpec>, ParameterGroupError>>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| {
                    ParameterGroupError::Metadata(WorkspaceMetadataError::Overflow.into())
                })?;
            context
                .charge_metadata(bytes)
                .map_err(|cause| ParameterGroupError::Metadata(cause.into()))?;
            context
                .metadata_vec(count)
                .map_err(ParameterGroupError::Metadata)?
        }
        None => Vec::with_capacity(count),
    };
    groups.push(group::<B::Tensor, _>(
        format_args!("{parameter_root}.embed_tokens"),
        ParameterRole::Vocabulary,
        embeddings,
        ShapePolicy::Embedding,
        context,
    )?);
    groups.push(group::<B::Tensor, _>(
        format_args!("{parameter_root}.norm"),
        ParameterRole::Replicated,
        norm,
        ShapePolicy::Replicated,
        context,
    )?);
    if let Some(head) = head {
        groups.push(group::<B::Tensor, _>(
            format_args!("lm_head"),
            ParameterRole::Vocabulary,
            head,
            ShapePolicy::Head,
            context,
        )?);
    }
    Ok(groups)
}

/// The actual metadata destination carried through one shared declaration worker.
#[derive(Clone, Copy)]
pub(crate) struct DeclarationDestination<'a>(pub(crate) Option<&'a WorkspaceContext>);

#[derive(Clone, Copy)]
pub(crate) enum NormalizationName {
    Block,
    Attention,
    FeedForward,
}
/// The exact actual declaration borrowed by either parameter visitor.
#[derive(Clone, Copy)]
pub(crate) enum DeclarationParameter<'a> {
    Ordinary(&'a eredu_nn::ParameterMetadata),
    Borrowed(eredu_nn::ParameterMetadataView<'a>),
}
impl<'a> DeclarationParameter<'a> {
    pub(crate) fn id(self) -> &'a eredu_nn::ParameterId {
        match self {
            Self::Ordinary(value) => &value.id,
            Self::Borrowed(value) => value.id(),
        }
    }
    pub(crate) fn companion(self) -> Option<eredu_nn::LinearCompanionRole> {
        match self {
            Self::Ordinary(value) => value.linear_companion,
            Self::Borrowed(value) => value.linear_companion(),
        }
    }
    pub(crate) fn companion_of(self) -> Option<&'a eredu_nn::ParameterId> {
        match self {
            Self::Ordinary(value) => value.linear_companion_of.as_ref(),
            Self::Borrowed(value) => value.linear_companion_of(),
        }
    }
}

impl DeclarationDestination<'_> {
    pub(crate) fn normalization_name(
        self,
        config: &impl Config,
        layer: usize,
        kind: NormalizationName,
        has_module: bool,
    ) -> Result<Option<String>, ParameterGroupError> {
        if let Some(context) = self.0 {
            if !has_module {
                return Ok(None);
            }
            return match kind {
                NormalizationName::Block => {
                    config.block_output_normalization_with_metadata(layer, context)
                }
                NormalizationName::Attention => {
                    config.attention_output_normalization_with_metadata(layer, context)
                }
                NormalizationName::FeedForward => {
                    config.feed_forward_output_normalization_with_metadata(layer, context)
                }
            }
            .map_err(ParameterGroupError::Metadata);
        }
        Ok(match kind {
            NormalizationName::Block => config.block_output_normalization(layer),
            NormalizationName::Attention => config.attention_output_normalization(layer),
            NormalizationName::FeedForward => config.feed_forward_output_normalization(layer),
        })
    }
    pub(crate) fn linear_format(
        self,
        config: &impl Config,
        name: &str,
    ) -> Result<LinearFormat, ParameterGroupError> {
        match self.0 {
            Some(context) => config
                .linear_format_with_metadata(name, context)
                .map_err(ParameterGroupError::Metadata),
            None => Ok(config.linear_format(name)),
        }
    }
    pub(crate) fn group_error(self, args: std::fmt::Arguments<'_>) -> ParameterGroupError {
        match self.0 {
            Some(context) => ParameterGroupError::Metadata(context.metadata_error(args)),
            None => {
                ParameterGroupError::Ordinary(ParallelPlanError::InvalidGroup(args.to_string()))
            }
        }
    }
    pub(crate) fn tensor_error(self, args: std::fmt::Arguments<'_>) -> ParameterGroupError {
        match self.0 {
            Some(context) => ParameterGroupError::Metadata(context.metadata_error(args)),
            None => {
                ParameterGroupError::Ordinary(ParallelPlanError::InvalidTensor(args.to_string()))
            }
        }
    }
    pub(crate) fn text(self, args: std::fmt::Arguments<'_>) -> Result<String, ParameterGroupError> {
        match self.0 {
            Some(context) => context
                .metadata_string(args)
                .map_err(ParameterGroupError::Metadata),
            None => Ok(args.to_string()),
        }
    }
    pub(crate) fn vector<T>(self, capacity: usize) -> Result<Vec<T>, ParameterGroupError> {
        match self.0 {
            Some(context) => context
                .metadata_vec(capacity)
                .map_err(ParameterGroupError::Metadata),
            None => Ok(Vec::with_capacity(capacity)),
        }
    }
    pub(crate) fn reserve<T>(
        self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), ParameterGroupError> {
        if let Some(context) = self.0 {
            context
                .reserve_metadata_vec(values, additional)
                .map_err(ParameterGroupError::Metadata)?;
        } else {
            values.reserve(additional);
        }
        Ok(())
    }
    pub(crate) fn controls<T>(self) -> Result<(), ParameterGroupError> {
        if let Some(context) = self.0 {
            let parts = [
                size_of::<T>(),
                size_of::<Self>(),
                size_of::<Result<T, ParameterGroupError>>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| {
                    ParameterGroupError::Metadata(WorkspaceMetadataError::Overflow.into())
                })?;
            context
                .charge_metadata(bytes)
                .map_err(|cause| ParameterGroupError::Metadata(cause.into()))?;
        }
        Ok(())
    }
    pub(crate) fn module<T: Tensor, M: eredu_nn::Parameterized<T>>(
        self,
        name: std::fmt::Arguments<'_>,
        role: ParameterRole,
        module: &M,
        mut sharding: impl FnMut(&[usize]) -> Result<MemberSharding, ParameterGroupError>,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        match self.0 {
            Some(context) => eredu_runtime::module_parameter_group_with_metadata::<T, M>(
                name,
                role,
                module,
                context,
                |_, shape| sharding(shape).map_err(ParameterGroupError::into_neural),
            )
            .map_err(ParameterGroupError::Metadata),
            None => module_parameter_group::<T, M>(name.to_string(), role, module, |_, shape| {
                sharding(shape).map_err(ParameterGroupError::ordinary)
            })
            .map_err(ParameterGroupError::Ordinary),
        }
    }
    pub(crate) fn partitioned_module<T: Tensor, M: eredu_nn::Parameterized<T>>(
        self,
        name: std::fmt::Arguments<'_>,
        role: ParameterRole,
        units: usize,
        module: &M,
        mut sharding: impl FnMut(&[usize]) -> Result<MemberSharding, ParameterGroupError>,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        self.partitioned_module_named::<T, M>(name, role, units, module, |_, shape| sharding(shape))
    }
    pub(crate) fn partitioned_module_named<T: Tensor, M: eredu_nn::Parameterized<T>>(
        self,
        name: std::fmt::Arguments<'_>,
        role: ParameterRole,
        units: usize,
        module: &M,
        mut sharding: impl FnMut(&str, &[usize]) -> Result<MemberSharding, ParameterGroupError>,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        self.partitioned_module_metadata::<T, M>(name, role, units, module, |metadata, shape| {
            sharding(metadata.id().as_str(), shape)
        })
    }
    pub(crate) fn partitioned_module_metadata<T: Tensor, M: eredu_nn::Parameterized<T>>(
        self,
        name: std::fmt::Arguments<'_>,
        role: ParameterRole,
        units: usize,
        module: &M,
        mut sharding: impl FnMut(
            DeclarationParameter<'_>,
            &[usize],
        ) -> Result<MemberSharding, ParameterGroupError>,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        self.controls::<(DeclarationParameter<'_>, MemberSharding)>()?;
        match self.0 {
            Some(context) => {
                eredu_runtime::partitioned_module_parameter_group_with_metadata::<T, M>(
                    name,
                    role,
                    units,
                    module,
                    context,
                    |metadata, shape| {
                        sharding(DeclarationParameter::Borrowed(metadata), shape)
                            .map_err(ParameterGroupError::into_neural)
                    },
                )
                .map_err(ParameterGroupError::Metadata)
            }
            None => partitioned_module_parameter_group::<T, M>(
                name.to_string(),
                role,
                units,
                module,
                |metadata, shape| {
                    sharding(DeclarationParameter::Ordinary(metadata), shape)
                        .map_err(ParameterGroupError::ordinary)
                },
            )
            .map_err(ParameterGroupError::Ordinary),
        }
    }
    pub(crate) fn aligned(
        self,
        name: &str,
        units: usize,
        width: usize,
        alignment: usize,
    ) -> Result<usize, ParameterGroupError> {
        match self.0 {
            Some(context) => eredu_runtime::aligned_partition_units_with_metadata(
                name, units, width, alignment, false, context,
            )
            .map_err(ParameterGroupError::Metadata),
            None => eredu_runtime::aligned_partition_units(name, units, width, alignment)
                .map_err(ParameterGroupError::Ordinary),
        }
    }
    pub(crate) fn projections<T: Tensor, M: eredu_nn::Parameterized<T>>(
        self,
        name: std::fmt::Arguments<'_>,
        role: ParameterRole,
        projections: &[(&M, ProjectionSharding)],
        units: usize,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        match self.0 {
            Some(context) => eredu_runtime::partitioned_projection_group_with_metadata::<T, M>(
                name,
                role,
                projections,
                units,
                context,
            )
            .map_err(ParameterGroupError::Metadata),
            None => {
                partitioned_projection_group::<T, M>(name.to_string(), role, projections, units)
                    .map_err(ParameterGroupError::Ordinary)
            }
        }
    }
    pub(crate) fn segmented<T: Tensor, M: eredu_nn::Parameterized<T>>(
        self,
        name: std::fmt::Arguments<'_>,
        role: ParameterRole,
        fused: &M,
        row: &M,
        segments: Vec<Range<usize>>,
        units: usize,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        match self.0 {
            Some(context) => eredu_runtime::segmented_projection_group_with_metadata::<T, M>(
                name, role, fused, row, segments, units, context,
            )
            .map_err(ParameterGroupError::Metadata),
            None => segmented_projection_group::<T, M>(
                name.to_string(),
                role,
                fused,
                row,
                segments,
                units,
            )
            .map_err(ParameterGroupError::Ordinary),
        }
    }
    pub(crate) fn repartition(
        self,
        group: ParameterGroupSpec,
        units: usize,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        match self.0 {
            Some(context) => group
                .into_partitioned_with_metadata(units, context)
                .map_err(ParameterGroupError::Metadata),
            None => group
                .into_partitioned(units)
                .map_err(ParameterGroupError::Ordinary),
        }
    }
    pub(crate) fn chunks(
        self,
        group: ParameterGroupSpec,
        units: usize,
        mut width: impl FnMut(
            &eredu_runtime::ParameterMemberSpec,
            &[eredu_runtime::ParameterMemberSpec],
        ) -> Result<usize, ParameterGroupError>,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        match self.0 {
            Some(context) => eredu_runtime::partition_parameter_group_chunks_with_metadata(
                group,
                units,
                context,
                |member, source| width(member, source).map_err(ParameterGroupError::into_neural),
            )
            .map_err(ParameterGroupError::Metadata),
            None => eredu_runtime::partition_parameter_group_chunks_with_source(
                group,
                units,
                |member, source| width(member, source).map_err(ParameterGroupError::ordinary),
            )
            .map_err(ParameterGroupError::Ordinary),
        }
    }
}

/// One indexed parameter identity with only borrowed semantic name components.
#[derive(Clone, Copy)]
pub(crate) struct LayerParameterName<'a> {
    pub(crate) root: &'a str,
    pub(crate) layer: usize,
    pub(crate) module: Option<&'a str>,
    pub(crate) field: &'a str,
}
impl std::fmt::Display for LayerParameterName<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.layers.{}.", self.root, self.layer)?;
        if let Some(module) = self.module {
            write!(f, "{module}.")?;
        }
        write!(f, "{}.weight", self.field)
    }
}
pub(crate) fn default_attention_value_name<C: Config + ?Sized>(
    config: &C,
    layer: usize,
) -> LayerParameterName<'_> {
    let fields = config.block_parameter_fields();
    LayerParameterName {
        root: config.parameter_root(),
        layer,
        module: Some(fields.attention),
        field: fields.attention_value,
    }
}
pub(crate) fn default_attention_value_format_with_metadata<C: Config + ?Sized>(
    config: &C,
    layer: usize,
    context: &WorkspaceContext,
) -> Result<eredu_checkpoint::LinearFormat, Error> {
    let name = default_attention_value_name(config, layer);
    let name = context.metadata_string(format_args!("{name}"))?;
    config.linear_format_with_metadata(&name, context)
}
