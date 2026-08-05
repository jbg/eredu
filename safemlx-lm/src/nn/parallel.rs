//! Reusable neural-network primitives with explicit tensor-parallel semantics.
//!
//! These modules do not retain communication groups. Callers borrow a
//! [`ParallelExecutionContext`] for each operation, allowing the same module
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
    error::Error,
    nn::{layers::silu, linear},
    runtime::{
        checkpoint::quantization::WeightQuantization,
        distributed::{
            parallel::{
                register_projection_module, register_replicated_module, MemberSharding,
                ParallelBuildContext, ParallelExecutionContext, ParallelPlanBuilder,
                ParameterGroupSpec, ParameterMemberSpec, ParameterRole, ProjectionSharding,
                ShardingPolicy,
            },
            topology::balanced_contiguous_range,
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
        if global_input_dims <= 0 || global_output_dims <= 0 {
            return Err(Error::Parallel(format!(
                "parallel linear dimensions must be positive, got {global_input_dims} -> {global_output_dims}"
            )));
        }
        let parts = context.topology().tensor_parallel_size;
        let dimension = match requested {
            LinearParallelism::Replicated => None,
            LinearParallelism::Column => Some(global_output_dims),
            LinearParallelism::Row => Some(global_input_dims),
        };
        let sharding_error = dimension.and_then(|dimension| {
            let dimension = usize::try_from(dimension).ok()?;
            let invalid_division = parts == 0 || dimension % parts != 0;
            let invalid_quantization = requested == LinearParallelism::Row
                && quantization.is_some_and(|quantization| {
                    let local = dimension.checked_div(parts).unwrap_or(0);
                    local == 0
                        || local % usize::try_from(quantization.group_size()).unwrap_or(usize::MAX)
                            != 0
                });
            (invalid_division || invalid_quantization).then_some(())
        });
        let (parallelism, fell_back_to_replication) = if sharding_error.is_some() {
            match context.policy() {
                ShardingPolicy::Require => {
                    return Err(Error::Parallel(format!(
                        "{requested:?} linear {global_input_dims} -> {global_output_dims} is incompatible with TP size {parts}{}",
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
        let local_input_dims = match parallelism {
            LinearParallelism::Row => {
                global_input_dims
                    / i32::try_from(parts)
                        .map_err(|_| Error::Parallel("TP size exceeds i32".into()))?
            }
            _ => global_input_dims,
        };
        let local_output_dims = match parallelism {
            LinearParallelism::Column => {
                global_output_dims
                    / i32::try_from(parts)
                        .map_err(|_| Error::Parallel("TP size exceeds i32".into()))?
            }
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

    /// Returns whether construction replicated an unsupported projection.
    pub const fn fell_back_to_replication(&self) -> bool {
        self.fell_back_to_replication
    }

    /// Describes every physical parameter without checkpoint-name inference.
    pub fn parameter_group(&self, prefix: &str) -> Result<ParameterGroupSpec, Error> {
        let (role, weight_sharding, output_companion) = match self.parallelism {
            LinearParallelism::Replicated => (
                ParameterRole::Replicated,
                MemberSharding::Replicated,
                MemberSharding::Replicated,
            ),
            LinearParallelism::Column => (
                ParameterRole::ColumnProjection,
                MemberSharding::Equal { axis: 0 },
                MemberSharding::Equal { axis: 0 },
            ),
            LinearParallelism::Row => (
                ParameterRole::RowProjection,
                MemberSharding::Equal { axis: 1 },
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
                    let local = usize::try_from(linear.inner.weight.value.dim(1)).unwrap();
                    if self.parallelism == LinearParallelism::Row {
                        local * self.tensor_parallel_size.max(1)
                    } else {
                        local
                    }
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
                        input_or_output_sharding(self.parallelism),
                    ));
                    if linear.biases.value.is_some() {
                        members.push(ParameterMemberSpec::new(
                            format!("{prefix}.biases"),
                            [output, input / usize::try_from(linear.group_size).unwrap()],
                            input_or_output_sharding(self.parallelism),
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
        ParameterGroupSpec::new(prefix, role, members)
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

fn input_or_output_sharding(parallelism: LinearParallelism) -> MemberSharding {
    match parallelism {
        LinearParallelism::Replicated => MemberSharding::Replicated,
        LinearParallelism::Column => MemberSharding::Equal { axis: 0 },
        LinearParallelism::Row => MemberSharding::Equal { axis: 1 },
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
            Err(error) => return Err(error),
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
            Err(error) => return Err(error),
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
        let parts = i32::try_from(context.topology().tensor_parallel_size)
            .map_err(|_| Error::Parallel("TP size exceeds i32".into()))?;
        let row_width = query_heads
            .checked_mul(value_head_dim)
            .ok_or_else(|| Error::Parallel("attention output width overflowed i32".into()))?;
        let shardable = query_heads > 0
            && kv_heads > 0
            && parts > 0
            && query_heads % parts == 0
            && kv_heads % parts == 0
            && row_width % parts == 0
            && quantization
                .is_none_or(|quantization| (row_width / parts) % quantization.group_size() == 0);
        let parallelism = if shardable {
            LinearParallelism::Column
        } else if context.policy() == ShardingPolicy::ReplicateUnsupported {
            LinearParallelism::Replicated
        } else {
            return Err(Error::Parallel(format!(
                "attention geometry q={query_heads}, kv={kv_heads}, value_head_dim={value_head_dim} is incompatible with TP size {parts}{}",
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
        let local_query_heads = if shardable {
            query_heads / parts
        } else {
            query_heads
        };
        let local_kv_heads = if shardable {
            kv_heads / parts
        } else {
            kv_heads
        };
        Ok(Self {
            q_proj: ParallelLinear::unloaded(
                hidden_size,
                query_heads * query_key_head_dim,
                bias,
                quantization,
                parallelism,
                context,
                stream,
            )?,
            k_proj: ParallelLinear::unloaded(
                hidden_size,
                kv_heads * query_key_head_dim,
                bias,
                quantization,
                parallelism,
                context,
                stream,
            )?,
            v_proj: ParallelLinear::unloaded(
                hidden_size,
                kv_heads * value_head_dim,
                bias,
                quantization,
                parallelism,
                context,
                stream,
            )?,
            o_proj: ParallelLinear::unloaded(
                query_heads * value_head_dim,
                hidden_size,
                bias,
                quantization,
                output_parallelism,
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
        let parts = i32::try_from(context.topology().tensor_parallel_size)
            .map_err(|_| Error::Parallel("TP size exceeds i32".into()))?;
        let shardable = intermediate_size > 0
            && parts > 0
            && intermediate_size % parts == 0
            && quantization.is_none_or(|quantization| {
                (intermediate_size / parts) % quantization.group_size() == 0
            });
        let (input_parallelism, output_parallelism) = if shardable {
            (LinearParallelism::Column, LinearParallelism::Row)
        } else if context.policy() == ShardingPolicy::ReplicateUnsupported {
            (LinearParallelism::Replicated, LinearParallelism::Replicated)
        } else {
            return Err(Error::Parallel(format!(
                "SwiGLU intermediate size {intermediate_size} is incompatible with TP size {parts}{}",
                quantization.map_or(String::new(), |quantization| format!(
                    " and quantization group size {}",
                    quantization.group_size()
                ))
            )));
        };
        Ok(Self {
            gate_proj: ParallelLinear::unloaded(
                hidden_size,
                intermediate_size,
                bias,
                quantization,
                input_parallelism,
                context,
                stream,
            )?,
            up_proj: ParallelLinear::unloaded(
                hidden_size,
                intermediate_size,
                bias,
                quantization,
                input_parallelism,
                context,
                stream,
            )?,
            down_proj: ParallelLinear::unloaded(
                intermediate_size,
                hidden_size,
                bias,
                quantization,
                output_parallelism,
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
    use crate::runtime::distributed::topology::{DeviceAssignment, ParallelTopology};
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
            ParallelTopology::from_rank(
                parts,
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
        assert!(error.to_string().contains("group size"));
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
    fn swiglu_fallback_replicates_the_complete_subgraph() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let mlp = ParallelSwiGluMlp::unloaded(
            8,
            10,
            false,
            None,
            build_context_with_policy(3, 0, ShardingPolicy::ReplicateUnsupported),
            &stream,
        )
        .unwrap();
        assert_eq!(mlp.gate_proj.parallelism(), LinearParallelism::Replicated);
        assert_eq!(mlp.up_proj.parallelism(), LinearParallelism::Replicated);
        assert_eq!(mlp.down_proj.parallelism(), LinearParallelism::Replicated);
    }
}
