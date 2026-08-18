//! Reusable neural-network primitives with explicit tensor-parallel semantics.
//!
//! These modules do not retain communication groups. Callers borrow a
//! [`crate::runtime::distributed::parallel::ParallelExecutionContext`] for each
//! operation, allowing the same module
//! implementation to execute in replicated and tensor-parallel modes.

use std::ops::Range;

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters as ModuleParametersTrait},
    nn,
    ops::zeros_like,
    quantization::MaybeQuantized,
    Array, Stream,
};

use crate::{
    core::balanced_contiguous_range,
    error::Error,
    nn::{convolution::DepthwiseConv1d, layers::silu, linear},
    runtime::{
        checkpoint::quantization::WeightQuantization,
        distributed::parallel::{
            aligned_partition_units, partitioned_projection_members,
            register_partitioned_projection_group, register_projection_module,
            register_replicated_module, MemberSharding, ParallelBuildContext,
            ParallelExecutionContext, ParallelPlanBuilder, ParameterGroupSpec, ParameterMemberSpec,
            ParameterRole, ProjectionSharding, ShardingPolicy,
        },
    },
};

/// Execution contract of one linear projection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LinearParallelism {
    /// Complete projection on every rank.
    Replicated,
    /// Output features are rank-local and are not gathered automatically.
    Column,
    /// Input features are rank-local and output partials are summed.
    Row,
}

/// Registers an architecture-native linear and all of its quantization
/// companions with the shared typed placement rules.
pub(crate) fn register_linear_parameter_group(
    planner: &mut ParallelPlanBuilder,
    projection: &MaybeQuantized<nn::Linear>,
    prefix: &str,
    parallelism: LinearParallelism,
) -> Result<(), Error> {
    let placement = match parallelism {
        LinearParallelism::Replicated => ProjectionSharding::Replicated,
        LinearParallelism::Column => ProjectionSharding::Column,
        LinearParallelism::Row => ProjectionSharding::Row,
    };
    register_projection_module(planner, projection, prefix, placement)
}

/// Registers all parameters in an architecture-owned module as replicated.
pub(crate) fn register_replicated_parameter_group(
    planner: &mut ParallelPlanBuilder,
    module: &impl ModuleParametersTrait,
    prefix: &str,
) -> Result<(), Error> {
    register_replicated_module(planner, module, prefix)
}

fn row_partition_alignment(projection: &MaybeQuantized<nn::Linear>) -> Result<usize, Error> {
    match projection {
        MaybeQuantized::Original(_) => Ok(1),
        MaybeQuantized::Quantized(projection) => usize::try_from(projection.group_size)
            .map_err(|_| Error::Parallel("projection quantization group exceeds usize".into())),
    }
}

/// Checkpoint-facing names for the four projections that share one GQA head
/// partition. Keeping names in the descriptor lets architecture modules reuse
/// the same semantic planner without pretending their tensor catalogs match.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GqaProjectionNames {
    pub query: &'static str,
    pub key: &'static str,
    pub value: &'static str,
    pub output: &'static str,
}

/// Checkpoint-facing names for projections sharing one SwiGLU intermediate
/// partition.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SwiGluProjectionNames {
    pub gate: &'static str,
    pub up: &'static str,
    pub down: &'static str,
}

/// Registers separate GQA projections as one logical head domain. The output
/// projection's packed-input alignment determines the smallest legal group of
/// complete KV/query-head bundles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_gqa_projection_group(
    planner: &mut ParallelPlanBuilder,
    prefix: &str,
    names: GqaProjectionNames,
    q_proj: &MaybeQuantized<nn::Linear>,
    k_proj: &MaybeQuantized<nn::Linear>,
    v_proj: &MaybeQuantized<nn::Linear>,
    o_proj: &MaybeQuantized<nn::Linear>,
    query_heads: i32,
    kv_heads: i32,
    head_dim: i32,
) -> Result<(), Error> {
    let (units, members) = gqa_projection_members(
        prefix,
        names,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        query_heads,
        kv_heads,
        head_dim,
    )?;
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.projections"),
        ParameterRole::AttentionHeads,
        units,
        members,
    )?)
}

/// Builds the physical members for one GQA head domain. Architectures with
/// additional per-head state, such as learned attention sinks, can append that
/// state before registering the resulting atomic parameter group.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_projection_members(
    prefix: &str,
    names: GqaProjectionNames,
    q_proj: &MaybeQuantized<nn::Linear>,
    k_proj: &MaybeQuantized<nn::Linear>,
    v_proj: &MaybeQuantized<nn::Linear>,
    o_proj: &MaybeQuantized<nn::Linear>,
    query_heads: i32,
    kv_heads: i32,
    head_dim: i32,
) -> Result<(usize, Vec<ParameterMemberSpec>), Error> {
    let query_heads = usize::try_from(query_heads)
        .map_err(|_| Error::Parallel("query-head count exceeds usize".into()))?;
    let kv_heads = usize::try_from(kv_heads)
        .map_err(|_| Error::Parallel("KV-head count exceeds usize".into()))?;
    let head_dim = usize::try_from(head_dim)
        .map_err(|_| Error::Parallel("attention head dimension exceeds usize".into()))?;
    if head_dim == 0 || kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
        return Err(Error::Parallel(format!(
            "attention head geometry q={query_heads}, kv={kv_heads}, dim={head_dim} does not form positive integral GQA groups"
        )));
    }
    let group_width = (query_heads / kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Parallel("GQA group width overflowed usize".into()))?;
    let units = aligned_partition_units(
        prefix,
        kv_heads,
        group_width,
        row_partition_alignment(o_proj)?,
    )?;
    let q_prefix = format!("{prefix}.{}", names.query);
    let k_prefix = format!("{prefix}.{}", names.key);
    let v_prefix = format!("{prefix}.{}", names.value);
    let o_prefix = format!("{prefix}.{}", names.output);
    partitioned_projection_members(
        &[
            (q_proj, q_prefix.as_str(), ProjectionSharding::Column),
            (k_proj, k_prefix.as_str(), ProjectionSharding::Column),
            (v_proj, v_prefix.as_str(), ProjectionSharding::Column),
            (o_proj, o_prefix.as_str(), ProjectionSharding::Row),
        ],
        units,
    )
}

/// Registers gate, up, and down projections as one logical dense
/// feed-forward intermediate domain, including packed quantization companions.
pub(crate) fn register_swiglu_projection_group(
    planner: &mut ParallelPlanBuilder,
    prefix: &str,
    names: SwiGluProjectionNames,
    gate_proj: &MaybeQuantized<nn::Linear>,
    up_proj: &MaybeQuantized<nn::Linear>,
    down_proj: &MaybeQuantized<nn::Linear>,
    intermediate_size: i32,
) -> Result<(), Error> {
    let intermediate = usize::try_from(intermediate_size)
        .map_err(|_| Error::Parallel("feed-forward width exceeds usize".into()))?;
    let units =
        aligned_partition_units(prefix, intermediate, 1, row_partition_alignment(down_proj)?)?;
    let gate_prefix = format!("{prefix}.{}", names.gate);
    let up_prefix = format!("{prefix}.{}", names.up);
    let down_prefix = format!("{prefix}.{}", names.down);
    let logical_name = format!("{prefix}.projections");
    register_partitioned_projection_group(
        planner,
        &logical_name,
        ParameterRole::FeedForwardIntermediate,
        &[
            (gate_proj, gate_prefix.as_str(), ProjectionSharding::Column),
            (up_proj, up_prefix.as_str(), ProjectionSharding::Column),
            (down_proj, down_prefix.as_str(), ProjectionSharding::Row),
        ],
        units,
    )
}

/// Registers a fused gated depthwise-convolution block as one logical channel
/// domain. The three input-projection segments, depthwise kernel, recurrent
/// channel state, and row projection therefore receive the identical range.
pub(crate) fn register_gated_depthwise_conv_group(
    planner: &mut ParallelPlanBuilder,
    prefix: &str,
    input_projection: &MaybeQuantized<nn::Linear>,
    convolution: &DepthwiseConv1d,
    output_projection: &MaybeQuantized<nn::Linear>,
    channels: i32,
) -> Result<(), Error> {
    let channels = usize::try_from(channels)
        .map_err(|_| Error::Parallel("convolution channel count exceeds usize".into()))?;
    let units = aligned_partition_units(
        prefix,
        channels,
        1,
        row_partition_alignment(output_projection)?,
    )?;
    let output_prefix = format!("{prefix}.out_proj");
    let (units, mut members) = partitioned_projection_members(
        &[(
            output_projection,
            output_prefix.as_str(),
            ProjectionSharding::Row,
        )],
        units,
    )?;
    let segments = vec![
        0..channels,
        channels..2 * channels,
        2 * channels..3 * channels,
    ];
    for (name, parameter) in input_projection.parameters().flatten() {
        let shape = parameter
            .shape()
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension).map_err(|_| {
                    Error::Parallel(format!(
                        "parameter {prefix}.in_proj.{name} has negative dimension {dimension}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape.first().copied() != Some(3 * channels) {
            return Err(Error::Parallel(format!(
                "parameter {prefix}.in_proj.{name} has shape {shape:?}, expected a fused three-segment channel axis of length {}",
                3 * channels
            )));
        }
        members.push(ParameterMemberSpec::new(
            format!("{prefix}.in_proj.{name}"),
            shape,
            MemberSharding::PartitionedSegments {
                axis: 0,
                segments: segments.clone(),
            },
        ));
    }
    for (name, parameter) in convolution.parameters().flatten() {
        let shape = parameter
            .shape()
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension).map_err(|_| {
                    Error::Parallel(format!(
                        "parameter {prefix}.conv.{name} has negative dimension {dimension}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape.first().copied() != Some(channels) {
            return Err(Error::Parallel(format!(
                "parameter {prefix}.conv.{name} has shape {shape:?}, expected channel axis length {channels}"
            )));
        }
        members.push(ParameterMemberSpec::new(
            format!("{prefix}.conv.{name}"),
            shape,
            MemberSharding::Partitioned { axis: 0 },
        ));
    }
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.channels"),
        ParameterRole::Channels,
        units,
        members,
    )?)
}

/// Returns the exact rank-local KV-head count selected for each decoder layer.
/// Cache creation and persistence use this planner-derived geometry instead of
/// reconstructing it from topology scalars.
pub(crate) fn planned_kv_head_layout(
    layout: &crate::runtime::distributed::parallel::LocalModelLayout,
    layer_count: usize,
    head_dim: i32,
    layer_prefix: &str,
) -> Result<Vec<i32>, Error> {
    planned_optional_kv_head_layout(
        layout,
        (0..layer_count).map(|_| true),
        head_dim,
        layer_prefix,
    )
    .map(|layers| {
        layers
            .into_iter()
            .map(|heads| heads.expect("all requested layers contain attention"))
            .collect()
    })
}

/// Returns planner-derived KV geometry for a heterogeneous layer schedule.
/// Layers without attention deliberately have no KV entry; their architecture
/// remains responsible for describing any other recurrent state.
pub(crate) fn planned_optional_kv_head_layout(
    layout: &crate::runtime::distributed::parallel::LocalModelLayout,
    attention_layers: impl IntoIterator<Item = bool>,
    head_dim: i32,
    layer_prefix: &str,
) -> Result<Vec<Option<i32>>, Error> {
    planned_optional_partition_widths(
        layout,
        attention_layers,
        head_dim,
        layer_prefix,
        "self_attn.k_proj",
    )
}

/// Returns the exact local semantic width for selected layers from one
/// planner-owned tensor axis. This is shared by attention heads and recurrent
/// or convolution channels so cache topology never re-derives partitions from
/// rank counts.
pub(crate) fn planned_optional_partition_widths(
    layout: &crate::runtime::distributed::parallel::LocalModelLayout,
    participating_layers: impl IntoIterator<Item = bool>,
    unit_width: i32,
    layer_prefix: &str,
    tensor_suffix: &str,
) -> Result<Vec<Option<i32>>, Error> {
    if unit_width <= 0 {
        return Err(Error::Parallel(format!(
            "partition unit width must be positive, got {unit_width}"
        )));
    }
    participating_layers
        .into_iter()
        .enumerate()
        .map(|(index, participates)| {
            if !participates {
                return Ok(None);
            }
            let prefix = format!("{layer_prefix}.{index}.{tensor_suffix}");
            let key = layout
                .tensor(&format!("{prefix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.inner.weight")))
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "missing {tensor_suffix} layout for decoder layer {index}"
                    ))
                })?;
            let local_width = i32::try_from(key.local_shape()[0])
                .map_err(|_| Error::Parallel("local partition width exceeds i32".into()))?;
            if local_width % unit_width != 0 {
                return Err(Error::Parallel(format!(
                    "local {tensor_suffix} width {local_width} splits semantic unit width {unit_width}"
                )));
            }
            let units = local_width / unit_width;
            if units <= 0 {
                return Err(Error::Parallel(format!(
                    "decoder layer {index} has no rank-local {tensor_suffix} units"
                )));
            }
            Ok(Some(units))
        })
        .collect()
}

/// Linear projection carrying its local geometry and collective contract.
#[derive(Debug, Clone)]
pub struct ParallelLinear {
    inner: MaybeQuantized<nn::Linear>,
    parallelism: LinearParallelism,
    requested_parallelism: LinearParallelism,
    global_input_dims: i32,
    global_output_dims: i32,
    local_input_dims: i32,
    local_output_dims: i32,
    tensor_parallel_size: usize,
    partition_units: Option<usize>,
    fell_back_to_replication: bool,
}

impl ModuleParametersTrait for ParallelLinear {
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        self.inner.parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.inner.parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.inner.trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.inner.freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.inner.unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        self.inner.all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        self.inner.any_frozen()
    }
}

impl ParallelLinear {
    /// Creates an unloaded dense or affine-quantized projection.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded(
        global_input_dims: i32,
        global_output_dims: i32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        requested: LinearParallelism,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let partition_units = match requested {
            LinearParallelism::Replicated => None,
            LinearParallelism::Column => {
                Some(usize::try_from(global_output_dims).map_err(|_| {
                    Error::Parallel("parallel linear output width exceeds usize".into())
                })?)
            }
            LinearParallelism::Row => {
                let input = usize::try_from(global_input_dims).map_err(|_| {
                    Error::Parallel("parallel linear input width exceeds usize".into())
                })?;
                let alignment = quantization.map_or(Ok(1usize), |quantization| {
                    usize::try_from(quantization.group_size()).map_err(|_| {
                        Error::Parallel("parallel linear quantization group exceeds usize".into())
                    })
                })?;
                Some(aligned_partition_units(
                    "row-parallel linear",
                    input,
                    1,
                    alignment,
                )?)
            }
        };
        Self::unloaded_with_partition_units(
            global_input_dims,
            global_output_dims,
            bias,
            quantization,
            requested,
            partition_units,
            context,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn unloaded_with_partition_units(
        global_input_dims: i32,
        global_output_dims: i32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        requested: LinearParallelism,
        requested_units: Option<usize>,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        if global_input_dims <= 0 || global_output_dims <= 0 {
            return Err(Error::Parallel(format!(
                "parallel linear dimensions must be positive, got {global_input_dims} -> {global_output_dims}"
            )));
        }
        let parts = context.topology().tensor_parallel_size;
        let dimension = match requested {
            LinearParallelism::Replicated => None,
            LinearParallelism::Column => Some(usize::try_from(global_output_dims).unwrap()),
            LinearParallelism::Row => Some(usize::try_from(global_input_dims).unwrap()),
        };
        let sharding_error = dimension
            .zip(requested_units)
            .is_some_and(|(dimension, units)| {
                parts == 0 || units < parts || units == 0 || !dimension.is_multiple_of(units)
            });
        let (parallelism, fell_back_to_replication) = if sharding_error {
            match context.policy() {
                ShardingPolicy::Require => {
                    return Err(Error::Parallel(format!(
                        "{requested:?} linear {global_input_dims} -> {global_output_dims} has no non-empty aligned partition for TP size {parts}{}",
                        quantization.map_or(String::new(), |quantization| format!(
                            " and quantization group size {}",
                            quantization.group_size()
                        ))
                    )))
                }
                ShardingPolicy::ReplicateUnsupported => (LinearParallelism::Replicated, true),
            }
        } else {
            (requested, false)
        };
        let partition_units = (!fell_back_to_replication)
            .then_some(requested_units)
            .flatten();
        let local_sharded_dimension = match (dimension, partition_units) {
            (Some(dimension), Some(units)) => {
                let logical = balanced_contiguous_range(
                    units,
                    parts,
                    context.topology().tensor_parallel_rank,
                    false,
                )?;
                logical.len() * (dimension / units)
            }
            _ => 0,
        };
        let local_input_dims = match parallelism {
            LinearParallelism::Row => i32::try_from(local_sharded_dimension)
                .map_err(|_| Error::Parallel("local row width exceeds i32".into()))?,
            _ => global_input_dims,
        };
        let local_output_dims = match parallelism {
            LinearParallelism::Column => i32::try_from(local_sharded_dimension)
                .map_err(|_| Error::Parallel("local column width exceeds i32".into()))?,
            _ => global_output_dims,
        };
        let inner = linear::unloaded_maybe_quantized_linear(
            local_input_dims,
            local_output_dims,
            bias,
            quantization,
            stream,
        )?;
        Ok(Self {
            inner,
            parallelism,
            requested_parallelism: requested,
            global_input_dims,
            global_output_dims,
            local_input_dims,
            local_output_dims,
            tensor_parallel_size: parts,
            partition_units,
            fell_back_to_replication,
        })
    }

    /// Returns the locally materialized projection.
    pub const fn inner(&self) -> &MaybeQuantized<nn::Linear> {
        &self.inner
    }

    /// Returns the mutable locally materialized projection.
    pub const fn inner_mut(&mut self) -> &mut MaybeQuantized<nn::Linear> {
        &mut self.inner
    }

    /// Returns the effective execution contract.
    pub const fn parallelism(&self) -> LinearParallelism {
        self.parallelism
    }

    /// Returns the originally requested execution contract.
    pub const fn requested_parallelism(&self) -> LinearParallelism {
        self.requested_parallelism
    }

    /// Returns local input width.
    pub const fn local_input_dims(&self) -> i32 {
        self.local_input_dims
    }

    /// Returns local output width.
    pub const fn local_output_dims(&self) -> i32 {
        self.local_output_dims
    }

    /// Returns every rank's output width for a column-sharded projection.
    pub fn column_output_widths(&self) -> Result<Vec<usize>, Error> {
        if self.parallelism != LinearParallelism::Column {
            return Err(Error::Parallel(format!(
                "output-width collection requires column parallelism, got {:?}",
                self.parallelism
            )));
        }
        let global = usize::try_from(self.global_output_dims)
            .map_err(|_| Error::Parallel("parallel linear output width is invalid".into()))?;
        let units = self.partition_units.ok_or_else(|| {
            Error::Parallel("column-parallel projection has no logical partition units".into())
        })?;
        if units == 0 || !global.is_multiple_of(units) {
            return Err(Error::Parallel(format!(
                "column-parallel output width {global} is incompatible with {units} logical units"
            )));
        }
        (0..self.tensor_parallel_size)
            .map(|rank| {
                balanced_contiguous_range(units, self.tensor_parallel_size, rank, false)
                    .map(|range| range.len() * (global / units))
                    .map_err(Error::from)
            })
            .collect()
    }

    /// Returns whether construction replicated an unsupported projection.
    pub const fn fell_back_to_replication(&self) -> bool {
        self.fell_back_to_replication
    }

    /// Describes every physical parameter without checkpoint-name inference.
    pub fn parameter_group(&self, prefix: &str) -> Result<ParameterGroupSpec, Error> {
        let aligned = |axis| {
            self.partition_units
                .map_or(MemberSharding::Replicated, |_| {
                    MemberSharding::Partitioned { axis }
                })
        };
        let (role, weight_sharding, output_companion) = match self.parallelism {
            LinearParallelism::Replicated => (
                ParameterRole::Replicated,
                MemberSharding::Replicated,
                MemberSharding::Replicated,
            ),
            LinearParallelism::Column => (ParameterRole::ColumnProjection, aligned(0), aligned(0)),
            LinearParallelism::Row => (
                ParameterRole::RowProjection,
                aligned(1),
                MemberSharding::Replicated,
            ),
        };
        let mut members = Vec::new();
        match &self.inner {
            MaybeQuantized::Original(linear) => {
                members.push(ParameterMemberSpec::new(
                    format!("{prefix}.weight"),
                    [
                        usize::try_from(self.global_output_dims).unwrap(),
                        usize::try_from(self.global_input_dims).unwrap(),
                    ],
                    weight_sharding,
                ));
                if linear.bias.value.is_some() {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.bias"),
                        [usize::try_from(self.global_output_dims).unwrap()],
                        output_companion,
                    ));
                }
            }
            MaybeQuantized::Quantized(linear) => {
                let input = usize::try_from(self.global_input_dims).unwrap();
                let output = usize::try_from(self.global_output_dims).unwrap();
                let native_iq = linear.native_format.is_some();
                let packed = if native_iq {
                    input / usize::try_from(linear.group_size).unwrap()
                        * usize::try_from(linear.bits).unwrap()
                } else {
                    usize::try_from(safemlx::ops::quantized_packed_dimension(
                        self.global_input_dims,
                        linear.bits,
                    ))
                    .unwrap()
                };
                members.push(ParameterMemberSpec::new(
                    format!("{prefix}.inner.weight"),
                    [output, packed],
                    weight_sharding,
                ));
                if !native_iq {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.scales"),
                        [output, input / usize::try_from(linear.group_size).unwrap()],
                        input_or_output_sharding(self.parallelism, self.partition_units),
                    ));
                    if linear.biases.value.is_some() {
                        members.push(ParameterMemberSpec::new(
                            format!("{prefix}.biases"),
                            [output, input / usize::try_from(linear.group_size).unwrap()],
                            input_or_output_sharding(self.parallelism, self.partition_units),
                        ));
                    }
                }
                if linear.inner.bias.value.is_some() {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.inner.bias"),
                        [output],
                        output_companion,
                    ));
                }
            }
        }
        match self.partition_units {
            Some(units) => ParameterGroupSpec::partitioned(prefix, role, units, members),
            None => ParameterGroupSpec::new(prefix, role, members),
        }
    }

    /// Executes the projection and its declared collective.
    pub fn forward(
        &mut self,
        input: &Array,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        self.validate_execution_context(context)?;
        match self.parallelism {
            LinearParallelism::Replicated | LinearParallelism::Column => {
                Ok(self.inner.forward(input, context.stream())?)
            }
            LinearParallelism::Row => {
                let partial =
                    forward_without_output_bias(&mut self.inner, input, context.stream())?;
                let reduced = context.all_sum(&partial)?;
                add_output_bias(&self.inner, reduced, context.stream())
            }
        }
    }

    fn validate_execution_context(
        &self,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<(), Error> {
        if self.parallelism != LinearParallelism::Replicated
            && (!context.is_tensor_parallel() || context.size() != self.tensor_parallel_size)
        {
            return Err(Error::Parallel(format!(
                "{:?} linear was built for TP size {} but executed with size {}",
                self.parallelism,
                self.tensor_parallel_size,
                context.size()
            )));
        }
        Ok(())
    }
}

fn input_or_output_sharding(
    parallelism: LinearParallelism,
    partition_units: Option<usize>,
) -> MemberSharding {
    let aligned = |axis| {
        partition_units.map_or(MemberSharding::Replicated, |_| {
            MemberSharding::Partitioned { axis }
        })
    };
    match parallelism {
        LinearParallelism::Replicated => MemberSharding::Replicated,
        LinearParallelism::Column => aligned(0),
        LinearParallelism::Row => aligned(1),
    }
}

fn forward_without_output_bias(
    projection: &mut MaybeQuantized<nn::Linear>,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    match projection {
        MaybeQuantized::Original(linear) => {
            safemlx::ops::matmul(input, linear.weight.value.transpose(stream)?, stream)
        }
        MaybeQuantized::Quantized(linear) => {
            let bias = linear.inner.bias.value.take();
            let result = linear.forward(input, stream);
            linear.inner.bias.value = bias;
            result
        }
    }
}

fn add_output_bias(
    projection: &MaybeQuantized<nn::Linear>,
    output: Array,
    stream: &Stream,
) -> Result<Array, Error> {
    let bias = match projection {
        MaybeQuantized::Original(linear) => linear.bias.value.as_ref(),
        MaybeQuantized::Quantized(linear) => linear.inner.bias.value.as_ref(),
    };
    match bias {
        Some(bias) => Ok(output.add(bias, stream)?),
        None => Ok(output),
    }
}

/// Executes a row-sharded projection stored in an architecture-native module.
/// Bias is applied once after the rank-local partials are reduced.
pub(crate) fn forward_row_parallel(
    projection: &mut MaybeQuantized<nn::Linear>,
    input: &Array,
    group: &safemlx::distributed::Group,
    stream: &Stream,
) -> Result<Array, Exception> {
    let partial = forward_without_output_bias(projection, input, stream)?;
    let output = safemlx::distributed::all_sum(&partial, group, stream)?;
    let bias = match projection {
        MaybeQuantized::Original(linear) => linear.bias.value.as_ref(),
        MaybeQuantized::Quantized(linear) => linear.inner.bias.value.as_ref(),
    };
    match bias {
        Some(bias) => output.add(bias, stream),
        None => Ok(output),
    }
}

/// Token embedding sharded by an uneven contiguous vocabulary range.
#[derive(Debug, Clone)]
pub struct VocabParallelEmbedding {
    inner: MaybeQuantized<nn::Embedding>,
    global_vocabulary: usize,
    dimensions: i32,
    range: Range<usize>,
    tensor_parallel_size: usize,
    replicated: bool,
}

impl ModuleParametersTrait for VocabParallelEmbedding {
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        self.inner.parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.inner.parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.inner.trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.inner.freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.inner.unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        self.inner.all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        self.inner.any_frozen()
    }
}

impl VocabParallelEmbedding {
    /// Creates an unloaded vocabulary-parallel embedding.
    pub fn unloaded(
        global_vocabulary: usize,
        dimensions: i32,
        quantization: Option<WeightQuantization>,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Self::unloaded_with_dtype(
            global_vocabulary,
            dimensions,
            quantization,
            safemlx::Dtype::Float32,
            context,
            stream,
        )
    }

    /// Creates an unloaded vocabulary-parallel embedding with an explicit
    /// dense checkpoint dtype.
    pub fn unloaded_with_dtype(
        global_vocabulary: usize,
        dimensions: i32,
        quantization: Option<WeightQuantization>,
        dense_dtype: safemlx::Dtype,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        if global_vocabulary == 0 || dimensions <= 0 {
            return Err(Error::Parallel(
                "vocabulary and embedding dimensions must be positive".into(),
            ));
        }
        let topology = context.topology();
        let range = balanced_contiguous_range(
            global_vocabulary,
            topology.tensor_parallel_size,
            topology.tensor_parallel_rank,
            false,
        );
        let (range, replicated) = match range {
            Ok(range) => (range, false),
            Err(_error) if context.policy() == ShardingPolicy::ReplicateUnsupported => {
                (0..global_vocabulary, true)
            }
            Err(error) => return Err(error.into()),
        };
        let inner = linear::unloaded_maybe_quantized_embedding_with_dtype(
            i32::try_from(range.len())
                .map_err(|_| Error::Parallel("local vocabulary exceeds i32".into()))?,
            dimensions,
            quantization,
            dense_dtype,
            stream,
        )?;
        Ok(Self {
            inner,
            global_vocabulary,
            dimensions,
            range,
            tensor_parallel_size: topology.tensor_parallel_size,
            replicated,
        })
    }

    /// Returns this rank's vocabulary ownership.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the locally materialized embedding.
    pub const fn inner(&self) -> &MaybeQuantized<nn::Embedding> {
        &self.inner
    }

    /// Returns the mutable locally materialized embedding.
    pub const fn inner_mut(&mut self) -> &mut MaybeQuantized<nn::Embedding> {
        &mut self.inner
    }

    /// Describes embedding parameters for typed checkpoint planning.
    pub fn parameter_group(&self, prefix: &str) -> Result<ParameterGroupSpec, Error> {
        vocab_embedding_parameter_group(
            &self.inner,
            prefix,
            self.global_vocabulary,
            self.dimensions,
            self.replicated,
        )
    }

    /// Embeds global token ids and reduces rank-local contributions.
    pub fn forward(
        &mut self,
        tokens: &Array,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        if self.replicated {
            return Ok(self.inner.forward(tokens, context.stream())?);
        }
        if context.size() != self.tensor_parallel_size || !context.is_tensor_parallel() {
            return Err(Error::Parallel(format!(
                "vocabulary embedding was built for TP size {} but executed with size {}",
                self.tensor_parallel_size,
                context.size()
            )));
        }
        let start = Array::from_int(
            i32::try_from(self.range.start)
                .map_err(|_| Error::Parallel("vocabulary start exceeds i32".into()))?,
        );
        let end = Array::from_int(
            i32::try_from(self.range.end)
                .map_err(|_| Error::Parallel("vocabulary end exceeds i32".into()))?,
        );
        let valid = tokens
            .ge(&start, context.stream())?
            .logical_and(tokens.lt(&end, context.stream())?, context.stream())?;
        let local_ids = tokens.subtract(&start, context.stream())?;
        let safe_ids =
            safemlx::ops::r#where(&valid, &local_ids, Array::from_int(0), context.stream())?;
        let local = self.inner.forward(&safe_ids, context.stream())?;
        let valid = valid.expand_dims(-1, context.stream())?;
        let local = safemlx::ops::r#where(
            &valid,
            &local,
            zeros_like(&local, context.stream())?,
            context.stream(),
        )?;
        context.all_sum(&local)
    }

    /// Projects hidden states with tied embedding weights into local logits.
    pub fn project_logits(
        &mut self,
        hidden: &Array,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<ShardedOutput, Error> {
        if !self.replicated
            && (context.size() != self.tensor_parallel_size || !context.is_tensor_parallel())
        {
            return Err(Error::Parallel(
                "vocabulary projection execution context does not match its build context".into(),
            ));
        }
        let array = match &mut self.inner {
            MaybeQuantized::Original(embedding) => embedding.as_linear(hidden, context.stream())?,
            MaybeQuantized::Quantized(embedding) => {
                embedding.as_linear(hidden, context.stream())?
            }
        };
        Ok(ShardedOutput {
            array,
            axis: -1,
            global_dimension: self.global_vocabulary,
            range: self.range.clone(),
            tensor_parallel_size: self.tensor_parallel_size,
            replicated: self.replicated,
        })
    }
}

/// Builds typed placement for a vocabulary embedding before its rank-local
/// module has been constructed.
pub fn vocab_embedding_parameter_group(
    inner: &MaybeQuantized<nn::Embedding>,
    prefix: &str,
    global_vocabulary: usize,
    dimensions: i32,
    replicated: bool,
) -> Result<ParameterGroupSpec, Error> {
    let role = if replicated {
        ParameterRole::Replicated
    } else {
        ParameterRole::Vocabulary
    };
    let sharding = if replicated {
        MemberSharding::Replicated
    } else {
        MemberSharding::Balanced { axis: 0 }
    };
    let mut members = Vec::new();
    let global = [global_vocabulary, usize::try_from(dimensions).unwrap()];
    match inner {
        MaybeQuantized::Original(_) => members.push(ParameterMemberSpec::new(
            format!("{prefix}.weight"),
            global,
            sharding,
        )),
        MaybeQuantized::Quantized(inner) => {
            let native_iq = inner.native_format.is_some();
            let packed = if native_iq {
                usize::try_from(inner.inner.weight.value.dim(1)).unwrap()
            } else {
                usize::try_from(safemlx::ops::quantized_packed_dimension(
                    dimensions, inner.bits,
                ))
                .unwrap()
            };
            members.push(ParameterMemberSpec::new(
                format!("{prefix}.inner.weight"),
                [global_vocabulary, packed],
                sharding.clone(),
            ));
            if !native_iq {
                members.push(ParameterMemberSpec::new(
                    format!("{prefix}.scales"),
                    [
                        global_vocabulary,
                        usize::try_from(dimensions / inner.group_size).unwrap(),
                    ],
                    sharding.clone(),
                ));
                if inner.biases.value.is_some() {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.biases"),
                        [
                            global_vocabulary,
                            usize::try_from(dimensions / inner.group_size).unwrap(),
                        ],
                        sharding,
                    ));
                }
            }
        }
    }
    ParameterGroupSpec::new(prefix, role, members)
}

/// Untied language-model head sharded by vocabulary rows.
#[derive(Debug, Clone)]
pub struct VocabParallelLmHead {
    inner: MaybeQuantized<nn::Linear>,
    global_input_dims: i32,
    global_vocabulary: usize,
    range: Range<usize>,
    tensor_parallel_size: usize,
    replicated: bool,
}

impl ModuleParametersTrait for VocabParallelLmHead {
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        self.inner.parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.inner.parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.inner.trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.inner.freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.inner.unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        self.inner.all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        self.inner.any_frozen()
    }
}

impl VocabParallelLmHead {
    /// Creates an unloaded uneven vocabulary head.
    pub fn unloaded(
        global_input_dims: i32,
        global_vocabulary: usize,
        quantization: Option<WeightQuantization>,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Self::unloaded_with_dtype(
            global_input_dims,
            global_vocabulary,
            quantization,
            safemlx::Dtype::Float32,
            context,
            stream,
        )
    }

    /// Creates an unloaded vocabulary head with an explicit dense checkpoint
    /// dtype.
    pub fn unloaded_with_dtype(
        global_input_dims: i32,
        global_vocabulary: usize,
        quantization: Option<WeightQuantization>,
        dense_dtype: safemlx::Dtype,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        if global_input_dims <= 0 || global_vocabulary == 0 {
            return Err(Error::Parallel(
                "language-model head dimensions must be positive".into(),
            ));
        }
        let topology = context.topology();
        let range = balanced_contiguous_range(
            global_vocabulary,
            topology.tensor_parallel_size,
            topology.tensor_parallel_rank,
            false,
        );
        let (range, replicated) = match range {
            Ok(range) => (range, false),
            Err(_error) if context.policy() == ShardingPolicy::ReplicateUnsupported => {
                (0..global_vocabulary, true)
            }
            Err(error) => return Err(error.into()),
        };
        let inner = linear::unloaded_maybe_quantized_linear_with_dtype(
            global_input_dims,
            i32::try_from(range.len())
                .map_err(|_| Error::Parallel("local vocabulary exceeds i32".into()))?,
            false,
            quantization,
            dense_dtype,
            stream,
        )?;
        Ok(Self {
            inner,
            global_input_dims,
            global_vocabulary,
            range,
            tensor_parallel_size: topology.tensor_parallel_size,
            replicated,
        })
    }

    /// Returns this rank's vocabulary ownership.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the locally materialized head.
    pub const fn inner_mut(&mut self) -> &mut MaybeQuantized<nn::Linear> {
        &mut self.inner
    }

    /// Describes output-head parameters for typed checkpoint planning.
    pub fn parameter_group(&self, prefix: &str) -> Result<ParameterGroupSpec, Error> {
        vocab_lm_head_parameter_group(
            &self.inner,
            prefix,
            self.global_input_dims,
            self.global_vocabulary,
            self.replicated,
        )
    }

    /// Computes local vocabulary logits without an implicit all-gather.
    pub fn forward(
        &mut self,
        hidden: &Array,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<ShardedOutput, Error> {
        if !self.replicated
            && (context.size() != self.tensor_parallel_size || !context.is_tensor_parallel())
        {
            return Err(Error::Parallel(
                "vocabulary head execution context does not match its build context".into(),
            ));
        }
        Ok(ShardedOutput {
            array: self.inner.forward(hidden, context.stream())?,
            axis: -1,
            global_dimension: self.global_vocabulary,
            range: self.range.clone(),
            tensor_parallel_size: self.tensor_parallel_size,
            replicated: self.replicated,
        })
    }
}

/// Builds typed placement for an untied vocabulary head before its rank-local
/// module has been constructed.
pub fn vocab_lm_head_parameter_group(
    inner: &MaybeQuantized<nn::Linear>,
    prefix: &str,
    global_input_dims: i32,
    global_vocabulary: usize,
    replicated: bool,
) -> Result<ParameterGroupSpec, Error> {
    let role = if replicated {
        ParameterRole::Replicated
    } else {
        ParameterRole::Vocabulary
    };
    let sharding = if replicated {
        MemberSharding::Replicated
    } else {
        MemberSharding::Balanced { axis: 0 }
    };
    let mut members = Vec::new();
    match inner {
        MaybeQuantized::Original(_) => members.push(ParameterMemberSpec::new(
            format!("{prefix}.weight"),
            [
                global_vocabulary,
                usize::try_from(global_input_dims).unwrap(),
            ],
            sharding,
        )),
        MaybeQuantized::Quantized(inner) => {
            let native_iq = inner.native_format.is_some();
            let packed = if native_iq {
                usize::try_from(inner.inner.weight.value.dim(1)).unwrap()
            } else {
                usize::try_from(safemlx::ops::quantized_packed_dimension(
                    global_input_dims,
                    inner.bits,
                ))
                .unwrap()
            };
            members.push(ParameterMemberSpec::new(
                format!("{prefix}.inner.weight"),
                [global_vocabulary, packed],
                sharding.clone(),
            ));
            if !native_iq {
                members.push(ParameterMemberSpec::new(
                    format!("{prefix}.scales"),
                    [
                        global_vocabulary,
                        usize::try_from(global_input_dims / inner.group_size).unwrap(),
                    ],
                    sharding.clone(),
                ));
                if inner.biases.value.is_some() {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.biases"),
                        [
                            global_vocabulary,
                            usize::try_from(global_input_dims / inner.group_size).unwrap(),
                        ],
                        sharding,
                    ));
                }
            }
        }
    }
    ParameterGroupSpec::new(prefix, role, members)
}

/// Array carrying its rank-local axis range.
#[derive(Debug)]
pub struct ShardedOutput {
    array: Array,
    axis: i32,
    global_dimension: usize,
    range: Range<usize>,
    tensor_parallel_size: usize,
    replicated: bool,
}

impl ShardedOutput {
    /// Returns the local array.
    pub const fn array(&self) -> &Array {
        &self.array
    }

    /// Consumes the wrapper and returns the local array.
    pub fn into_array(self) -> Array {
        self.array
    }

    /// Returns the sharded axis.
    pub const fn axis(&self) -> i32 {
        self.axis
    }

    /// Returns the complete axis width.
    pub const fn global_dimension(&self) -> usize {
        self.global_dimension
    }

    /// Returns this rank's global range.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Gathers the complete uneven axis on every rank.
    pub fn all_gather(&self, context: &ParallelExecutionContext<'_>) -> Result<Array, Error> {
        if self.replicated {
            return Ok(self.array.clone());
        }
        if context.size() != self.tensor_parallel_size {
            return Err(Error::Parallel(format!(
                "sharded output was built for TP size {} but gathered with size {}",
                self.tensor_parallel_size,
                context.size()
            )));
        }
        let group = context.group().ok_or_else(|| {
            Error::Parallel("sharded output requires a tensor-parallel execution context".into())
        })?;
        let widths = (0..context.size())
            .map(|rank| {
                balanced_contiguous_range(self.global_dimension, context.size(), rank, false)
                    .map(|range| range.len())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(safemlx::distributed::all_gather_uneven_axis(
            &self.array,
            self.axis,
            &widths,
            group,
            context.stream(),
        )?)
    }
}

/// Q/K/V column projections and row-parallel output projection.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ParallelAttentionProjections {
    #[param]
    /// Query projection.
    pub q_proj: ParallelLinear,
    #[param]
    /// Key projection.
    pub k_proj: ParallelLinear,
    #[param]
    /// Value projection.
    pub v_proj: ParallelLinear,
    #[param]
    /// Output projection.
    pub o_proj: ParallelLinear,
    global_query_heads: i32,
    global_kv_heads: i32,
    local_query_heads: i32,
    local_kv_heads: i32,
    fell_back_to_replication: bool,
}

impl ParallelAttentionProjections {
    /// Creates standard separate attention projections with local head geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded(
        hidden_size: i32,
        query_heads: i32,
        kv_heads: i32,
        query_key_head_dim: i32,
        value_head_dim: i32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let parts = context.topology().tensor_parallel_size;
        let row_width = query_heads
            .checked_mul(value_head_dim)
            .ok_or_else(|| Error::Parallel("attention output width overflowed i32".into()))?;
        let query_per_kv = (query_heads > 0 && kv_heads > 0 && query_heads % kv_heads == 0)
            .then_some(query_heads / kv_heads);
        let alignment = quantization.map_or(Ok(1usize), |quantization| {
            usize::try_from(quantization.group_size())
                .map_err(|_| Error::Parallel("attention quantization group exceeds usize".into()))
        })?;
        let partition_units = query_per_kv
            .and_then(|query_per_kv| {
                aligned_partition_units(
                    "parallel attention",
                    usize::try_from(kv_heads).ok()?,
                    usize::try_from(query_per_kv.checked_mul(value_head_dim)?).ok()?,
                    alignment,
                )
                .ok()
            })
            .filter(|units| *units >= parts && parts > 0);
        let shardable = partition_units.is_some();
        let parallelism = if shardable {
            LinearParallelism::Column
        } else if context.policy() == ShardingPolicy::ReplicateUnsupported {
            LinearParallelism::Replicated
        } else {
            return Err(Error::Parallel(format!(
                "attention geometry q={query_heads}, kv={kv_heads}, value_head_dim={value_head_dim} has no non-empty aligned partition for TP size {parts}{}",
                quantization.map_or(String::new(), |quantization| format!(
                    " and quantization group size {}",
                    quantization.group_size()
                ))
            )));
        };
        let output_parallelism = if parallelism == LinearParallelism::Column {
            LinearParallelism::Row
        } else {
            LinearParallelism::Replicated
        };
        let (local_query_heads, local_kv_heads) = if let Some(units) = partition_units {
            let logical = balanced_contiguous_range(
                units,
                parts,
                context.topology().tensor_parallel_rank,
                false,
            )?;
            let local_kv_heads = i32::try_from(
                logical.len()
                    * (usize::try_from(kv_heads)
                        .map_err(|_| Error::Parallel("KV heads exceed usize".into()))?
                        / units),
            )
            .map_err(|_| Error::Parallel("local KV heads exceed i32".into()))?;
            (
                local_kv_heads * query_per_kv.expect("validated GQA ratio"),
                local_kv_heads,
            )
        } else {
            (query_heads, kv_heads)
        };
        let projection_units = shardable.then_some(partition_units).flatten();
        Ok(Self {
            q_proj: ParallelLinear::unloaded_with_partition_units(
                hidden_size,
                query_heads * query_key_head_dim,
                bias,
                quantization,
                parallelism,
                projection_units,
                context,
                stream,
            )?,
            k_proj: ParallelLinear::unloaded_with_partition_units(
                hidden_size,
                kv_heads * query_key_head_dim,
                bias,
                quantization,
                parallelism,
                projection_units,
                context,
                stream,
            )?,
            v_proj: ParallelLinear::unloaded_with_partition_units(
                hidden_size,
                kv_heads * value_head_dim,
                bias,
                quantization,
                parallelism,
                projection_units,
                context,
                stream,
            )?,
            o_proj: ParallelLinear::unloaded_with_partition_units(
                row_width,
                hidden_size,
                bias,
                quantization,
                output_parallelism,
                projection_units,
                context,
                stream,
            )?,
            global_query_heads: query_heads,
            global_kv_heads: kv_heads,
            local_query_heads,
            local_kv_heads,
            fell_back_to_replication: !shardable,
        })
    }

    /// Returns global and local query-head counts.
    pub const fn query_heads(&self) -> (i32, i32) {
        (self.global_query_heads, self.local_query_heads)
    }

    /// Returns global and local key/value-head counts.
    pub const fn kv_heads(&self) -> (i32, i32) {
        (self.global_kv_heads, self.local_kv_heads)
    }

    /// Returns whether incompatible geometry replicated the complete block.
    pub const fn fell_back_to_replication(&self) -> bool {
        self.fell_back_to_replication
    }

    /// Computes rank-local query, key, and value projections.
    pub fn project_qkv(
        &mut self,
        hidden: &Array,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<(Array, Array, Array), Error> {
        Ok((
            self.q_proj.forward(hidden, context)?,
            self.k_proj.forward(hidden, context)?,
            self.v_proj.forward(hidden, context)?,
        ))
    }

    /// Projects local attended values and returns the reduced hidden delta.
    pub fn project_output(
        &mut self,
        attended: &Array,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        self.o_proj.forward(attended, context)
    }
}

/// SwiGLU MLP composed from column- and row-parallel projections.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ParallelSwiGluMlp {
    #[param]
    /// Gate projection.
    pub gate_proj: ParallelLinear,
    #[param]
    /// Up projection.
    pub up_proj: ParallelLinear,
    #[param]
    /// Down projection and reduction.
    pub down_proj: ParallelLinear,
    fell_back_to_replication: bool,
}

impl ParallelSwiGluMlp {
    /// Creates an unloaded parallel SwiGLU block.
    #[allow(clippy::too_many_arguments)]
    pub fn unloaded(
        hidden_size: i32,
        intermediate_size: i32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let parts = context.topology().tensor_parallel_size;
        let alignment = quantization.map_or(Ok(1usize), |quantization| {
            usize::try_from(quantization.group_size())
                .map_err(|_| Error::Parallel("MLP quantization group exceeds usize".into()))
        })?;
        let partition_units = usize::try_from(intermediate_size)
            .ok()
            .and_then(|intermediate| {
                aligned_partition_units("parallel SwiGLU", intermediate, 1, alignment).ok()
            })
            .filter(|units| *units >= parts && parts > 0);
        let shardable = partition_units.is_some();
        let (input_parallelism, output_parallelism) = if shardable {
            (LinearParallelism::Column, LinearParallelism::Row)
        } else if context.policy() == ShardingPolicy::ReplicateUnsupported {
            (LinearParallelism::Replicated, LinearParallelism::Replicated)
        } else {
            return Err(Error::Parallel(format!(
                "SwiGLU intermediate size {intermediate_size} has no non-empty aligned partition for TP size {parts}{}",
                quantization.map_or(String::new(), |quantization| format!(
                    " and quantization group size {}",
                    quantization.group_size()
                ))
            )));
        };
        Ok(Self {
            gate_proj: ParallelLinear::unloaded_with_partition_units(
                hidden_size,
                intermediate_size,
                bias,
                quantization,
                input_parallelism,
                partition_units,
                context,
                stream,
            )?,
            up_proj: ParallelLinear::unloaded_with_partition_units(
                hidden_size,
                intermediate_size,
                bias,
                quantization,
                input_parallelism,
                partition_units,
                context,
                stream,
            )?,
            down_proj: ParallelLinear::unloaded_with_partition_units(
                intermediate_size,
                hidden_size,
                bias,
                quantization,
                output_parallelism,
                partition_units,
                context,
                stream,
            )?,
            fell_back_to_replication: !shardable,
        })
    }

    /// Returns whether incompatible geometry replicated the complete MLP.
    pub const fn fell_back_to_replication(&self) -> bool {
        self.fell_back_to_replication
    }

    /// Executes local gate/up work and one row-parallel reduction.
    pub fn forward(
        &mut self,
        hidden: &Array,
        context: &ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let gate = silu(self.gate_proj.forward(hidden, context)?, context.stream())?;
        let up = self.up_proj.forward(hidden, context)?;
        let local = gate.multiply(up, context.stream())?;
        self.down_proj.forward(&local, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mlx::{DeviceAssignment, MlxParallelContext};
    use safemlx::{module::ModuleParameters, DeviceType};

    fn build_context(parts: usize, rank: usize) -> ParallelBuildContext {
        build_context_with_policy(parts, rank, ShardingPolicy::Require)
    }

    fn build_context_with_policy(
        parts: usize,
        rank: usize,
        policy: ShardingPolicy,
    ) -> ParallelBuildContext {
        ParallelBuildContext::new(
            MlxParallelContext::for_rank(
                rank,
                parts,
                1,
                1,
                DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap(),
            policy,
        )
    }

    #[test]
    fn parallel_linear_exposes_local_shapes_and_typed_roles() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let linear = ParallelLinear::unloaded(
            16,
            32,
            true,
            None,
            LinearParallelism::Column,
            build_context(2, 0),
            &stream,
        )
        .unwrap();
        assert_eq!(linear.local_input_dims(), 16);
        assert_eq!(linear.local_output_dims(), 16);
        let params = linear.parameters().flatten();
        assert_eq!(params["weight"].shape(), &[16, 16]);
        assert_eq!(params["bias"].shape(), &[16]);
        let group = linear.parameter_group("projection").unwrap();
        assert_eq!(group.role(), ParameterRole::ColumnProjection);
    }

    #[test]
    fn row_quantization_requires_local_group_alignment() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let quantization = WeightQuantization::Affine(
            crate::runtime::checkpoint::quantization::AffineQuantization::new(64, 4).unwrap(),
        );
        let error = ParallelLinear::unloaded(
            96,
            32,
            false,
            Some(quantization),
            LinearParallelism::Row,
            build_context(2, 0),
            &stream,
        )
        .unwrap_err();
        assert!(error.to_string().contains("alignment-64"));
    }

    #[test]
    fn row_quantization_balances_complete_groups() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let quantization = WeightQuantization::Affine(
            crate::runtime::checkpoint::quantization::AffineQuantization::new(64, 4).unwrap(),
        );
        let first = ParallelLinear::unloaded(
            192,
            32,
            false,
            Some(quantization),
            LinearParallelism::Row,
            build_context(2, 0),
            &stream,
        )
        .unwrap();
        let second = ParallelLinear::unloaded(
            192,
            32,
            false,
            Some(quantization),
            LinearParallelism::Row,
            build_context(2, 1),
            &stream,
        )
        .unwrap();
        assert_eq!(first.local_input_dims(), 128);
        assert_eq!(second.local_input_dims(), 64);
        let first = first.parameter_group("projection").unwrap();
        assert_eq!(first.partition_units(), Some(3));
        assert!(first
            .members()
            .iter()
            .all(|member| matches!(member.sharding(), MemberSharding::Partitioned { .. })));
    }

    #[test]
    fn vocabulary_modules_use_balanced_local_rows() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let embedding =
            VocabParallelEmbedding::unloaded(11, 8, None, build_context(3, 2), &stream).unwrap();
        assert_eq!(embedding.range(), 8..11);
        let parameters = embedding.parameters().flatten();
        assert_eq!(parameters["weight"].shape(), &[3, 8]);
    }

    #[test]
    fn sharded_output_rejects_a_replicated_gather_context() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let output = ShardedOutput {
            array: Array::from_int(1),
            axis: -1,
            global_dimension: 2,
            range: 0..1,
            tensor_parallel_size: 2,
            replicated: false,
        };
        let execution = ParallelExecutionContext::replicated(&stream);
        assert!(output.all_gather(&execution).is_err());
    }

    #[test]
    fn attention_fallback_replicates_the_complete_projection_set() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let attention = ParallelAttentionProjections::unloaded(
            16,
            8,
            2,
            2,
            2,
            false,
            None,
            build_context_with_policy(4, 0, ShardingPolicy::ReplicateUnsupported),
            &stream,
        )
        .unwrap();
        assert_eq!(attention.query_heads(), (8, 8));
        assert_eq!(attention.kv_heads(), (2, 2));
        for projection in [
            &attention.q_proj,
            &attention.k_proj,
            &attention.v_proj,
            &attention.o_proj,
        ] {
            assert_eq!(projection.parallelism(), LinearParallelism::Replicated);
        }
    }

    #[test]
    fn attention_balances_whole_gqa_groups() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let first = ParallelAttentionProjections::unloaded(
            12,
            6,
            3,
            2,
            2,
            true,
            None,
            build_context(2, 0),
            &stream,
        )
        .unwrap();
        let second = ParallelAttentionProjections::unloaded(
            12,
            6,
            3,
            2,
            2,
            true,
            None,
            build_context(2, 1),
            &stream,
        )
        .unwrap();
        assert_eq!(first.query_heads(), (6, 4));
        assert_eq!(first.kv_heads(), (3, 2));
        assert_eq!(second.query_heads(), (6, 2));
        assert_eq!(second.kv_heads(), (3, 1));
        assert_eq!(first.o_proj.local_input_dims(), 8);
        assert_eq!(second.o_proj.local_input_dims(), 4);
    }

    #[test]
    fn swiglu_balances_uneven_intermediate_widths() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let first =
            ParallelSwiGluMlp::unloaded(8, 10, false, None, build_context(3, 0), &stream).unwrap();
        let last =
            ParallelSwiGluMlp::unloaded(8, 10, false, None, build_context(3, 2), &stream).unwrap();
        assert_eq!(first.gate_proj.local_output_dims(), 4);
        assert_eq!(first.down_proj.local_input_dims(), 4);
        assert_eq!(last.gate_proj.local_output_dims(), 3);
        assert_eq!(last.down_proj.local_input_dims(), 3);
    }
}
