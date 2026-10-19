//! One context/proposal traversal for resident and completion-safe leased bodies.

use super::{
    append_projected_context, context_append_start, DFlashContext, DFlashLayerContext,
    ProposalLayout,
};
use crate::decoder::ComponentInstrumentation;
use eredu_nn::{
    DistributedNeuralBackend, Error, Index, LinearOperator, NeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::{
    ExecutionGraph, ExecutionUnitLayout, LayerwiseAcquireError, LayerwisePolicy,
    LayerwisePolicyForward,
};

mod dflash;
pub use dflash::{DFlashBodyDefinition, DFlashBodyUnit};
mod dspark;
pub use dspark::{DsparkBodyDefinition, DsparkBodyUnit};

/// A trained assistant projection consumes and returns its parameter compute dtype.
pub(super) fn parameter_projection<B: NeuralBackend>(
    projection: &mut B::Linear,
    input: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let dtype = projection
        .parameter_element_type()
        .ok_or_else(|| Error::backend("assistant projection has no declared compute dtype"))?;
    projection
        .forward(&input.cast_float(dtype, context)?, context)?
        .cast_float(dtype, context)
}

/// Transient sublayer output and separately rounded residual between block leases.
/// The residual is absent for bodies that combine their branches eagerly.
pub struct AssistantBodyState<T> {
    /// Current branch output used as the execution/completion prototype.
    pub hidden: T,
    /// Prior residual retained until the next fused normalization.
    pub residual: Option<T>,
}

/// Validated context, decoder and final-normalization acquisition geometry.
#[derive(Debug, Clone)]
pub struct AssistantBodyGeometry {
    pub(super) hidden: i32,
    pub(super) captures: i32,
    pub(super) layers: usize,
    key_value_heads: i32,
    head_dim: i32,
    pub(super) block: i32,
    pub(super) window: i32,
    pub(super) proposal: ProposalLayout,
    layout: ExecutionUnitLayout,
}
impl AssistantBodyGeometry {
    pub(super) fn new(
        hidden: i32,
        captures: usize,
        layers: usize,
        key_value_heads: i32,
        head_dim: i32,
        block: usize,
        window: i32,
        proposal: ProposalLayout,
    ) -> Result<Self, Error> {
        let captures = i32::try_from(captures).map_err(Error::backend)?;
        let block = i32::try_from(block).map_err(Error::backend)?;
        if hidden <= 0
            || captures <= 0
            || layers == 0
            || key_value_heads <= 0
            || head_dim <= 0
            || window <= 0
            || block
                < if proposal == ProposalLayout::AnchorThenMasks {
                    2
                } else {
                    1
                }
            || hidden.checked_mul(captures).is_none()
        {
            return Err(Error::backend("invalid assistant body geometry"));
        }
        let graph = ExecutionGraph::chain([
            "assistant.context",
            "assistant.decoder",
            "assistant.final_norm",
        ])
        .map_err(Error::backend)?;
        let layout = ExecutionUnitLayout::new(&graph, [1, layers, 1]).map_err(Error::backend)?;
        Ok(Self {
            hidden,
            captures,
            layers,
            key_value_heads,
            head_dim,
            block,
            window,
            proposal,
            layout,
        })
    }
    /// Canonical context/block/final-norm addresses, independent of storage mode.
    pub const fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.layout
    }
    /// Decoder and projected context width.
    pub const fn hidden_size(&self) -> i32 {
        self.hidden
    }
    /// Number of assistant decoder blocks.
    pub const fn layer_count(&self) -> usize {
        self.layers
    }
    fn validate_context<T: Tensor>(&self, committed: &DFlashContext<T>) -> Result<i32, Error> {
        let retained = committed
            .end
            .checked_sub(committed.start)
            .filter(|n| *n > 0 && *n <= self.window)
            .ok_or_else(|| Error::backend("invalid assistant context positions"))?;
        if committed.start < 0
            || committed.encoded.shape() != [1, retained, self.hidden]
            || committed.layers.len() != self.layers
            || committed.layers.iter().any(|p| {
                let expected = [1, self.key_value_heads, retained, self.head_dim];
                p.keys.shape() != expected || p.values.shape() != expected
            })
        {
            return Err(Error::backend(
                "invalid assistant committed context geometry",
            ));
        }
        Ok(retained)
    }
    /// Concatenates validated target publications in their admitted order.
    pub fn assemble_target_states<T: Tensor>(
        &self,
        states: &[T],
        context: &T::Context,
    ) -> Result<T, Error> {
        if states.len() != self.captures as usize
            || states.is_empty()
            || states.iter().any(|s| {
                s.shape().len() != 3
                    || s.shape()[..2] != states[0].shape()[..2]
                    || s.dim(2) != self.hidden
            })
        {
            return Err(Error::backend("invalid ordered assistant target states"));
        }
        T::concatenate(states, 2, context)
    }
}

/// Architecture equations over independently populated context, block and norm units.
/// Sampling, verification and publication remain in the ordinary speculative driver.
pub trait AssistantBodyDefinition<B: NeuralBackend> {
    /// One typed parameter unit; a unit never contains the complete assistant.
    type Unit: Parameterized<B::Tensor>;
    /// Validated shape and canonical acquisition layout.
    fn geometry(&self) -> &AssistantBodyGeometry;
    /// Constructs exactly one unloaded unit at a canonical flat ordinal.
    fn build_unit(
        &self,
        ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error>;
    /// Constructs the resident unit sequence using the same per-unit constructors.
    fn build_units(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<Self::Unit>, Error> {
        (0..self.geometry().unit_layout().len())
            .map(|ordinal| self.build_unit(ordinal, context))
            .collect()
    }
    /// Encodes newly committed target taps.
    fn encode(
        &self,
        unit: &mut Self::Unit,
        pending: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
    /// Projects one block's committed keys and values in the released norm order.
    fn project_context(
        &self,
        unit: &mut Self::Unit,
        encoded: &B::Tensor,
        start: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DFlashLayerContext<B::Tensor>, Error>;
    /// Prepares the architecture's residual precision and optional proposal mask.
    fn begin_proposal(
        &self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(AssistantBodyState<B::Tensor>, Option<B::Tensor>), Error>;
    /// Executes one block against committed context without mutating it.
    fn forward_block(
        &self,
        unit: &mut Self::Unit,
        state: &AssistantBodyState<B::Tensor>,
        prior: &DFlashLayerContext<B::Tensor>,
        retained: i32,
        end: i32,
        mask: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<AssistantBodyState<B::Tensor>, Error>;
    /// Applies the independently leased final normalization.
    fn normalize(
        &self,
        unit: &mut Self::Unit,
        state: &AssistantBodyState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}

/// Bodies whose local projections produce complete activations through TP sums.
/// Context encoding and unit leasing remain in the ordinary shared traversal.
pub trait TensorParallelAssistantBodyDefinition<B: NeuralBackend + DistributedNeuralBackend>:
    AssistantBodyDefinition<B>
{
    /// Executes one local block, reducing projections before nonlinear equations.
    fn forward_block_parallel(
        &self,
        unit: &mut Self::Unit,
        state: &AssistantBodyState<B::Tensor>,
        prior: &DFlashLayerContext<B::Tensor>,
        retained: i32,
        end: i32,
        mask: Option<&B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<AssistantBodyState<B::Tensor>, Error>;
}

fn acquire_error<P: std::error::Error + Send + Sync + 'static>(
    error: LayerwiseAcquireError<Error, P>,
) -> Error {
    match error {
        LayerwiseAcquireError::Architecture(e) => e,
        LayerwiseAcquireError::Policy(e) => Error::backend_source(e),
    }
}

/// Encodes committed context using the selected ordinary unit policy. Every
/// projected K/V tensor is passed to exact completion before its block is released.
pub fn update_context<B, D, P>(
    definition: &D,
    policy: &mut P,
    previous: Option<DFlashContext<B::Tensor>>,
    pending: &B::Tensor,
    end: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<DFlashContext<B::Tensor>, Error>
where
    B: NeuralBackend,
    D: AssistantBodyDefinition<B>,
    P: LayerwisePolicy<B, D::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    let g = definition.geometry();
    if let Some(previous) = &previous {
        g.validate_context(previous)?;
    }
    let length = pending.shape().get(1).copied().unwrap_or(0);
    let start = context_append_start(previous.as_ref().map(|p| p.end), length, end)?;
    if pending.shape() != [1, length, g.hidden * g.captures] {
        return Err(Error::backend("invalid assistant committed target taps"));
    }
    let mut forward = LayerwisePolicyForward::<B, D::Unit, P>::begin(policy, pending, context)
        .map_err(Error::backend_source)?;
    let unit = forward
        .acquire(0, g.layout.address(0).unwrap(), |ctx| {
            definition.build_unit(0, ctx)
        })
        .map_err(acquire_error)?;
    let encoded = definition.encode(unit, pending, context)?;
    forward
        .complete(&encoded, std::iter::empty(), std::iter::once(pending))
        .map_err(Error::backend_source)?;
    let mut projected = Vec::with_capacity(g.layers);
    for layer in 0..g.layers {
        let ordinal = layer + 1;
        let unit = forward
            .acquire(ordinal, g.layout.address(ordinal).unwrap(), |ctx| {
                definition.build_unit(ordinal, ctx)
            })
            .map_err(acquire_error)?;
        let kv = definition.project_context(unit, &encoded, start, context)?;
        forward
            .complete(
                &kv.keys,
                std::iter::once(&kv.values),
                std::iter::once(&encoded),
            )
            .map_err(Error::backend_source)?;
        projected.push(kv);
    }
    let result = append_projected_context(previous, encoded, projected, end, g.window, context)?;
    // Compaction owns its state inputs; weight-dependent K/V outputs have
    // already crossed each unit's completion boundary above.
    forward
        .finish(&result.encoded)
        .map_err(Error::backend_source)?;
    Ok(result)
}

/// Runs a proposal block through the same context, neural block and output
/// equations for every parameter policy. Failed execution aborts its active lease.
pub fn proposal_states<B, D, P>(
    definition: &D,
    policy: &mut P,
    embeddings: &B::Tensor,
    committed: &DFlashContext<B::Tensor>,
    end: i32,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend,
    D: AssistantBodyDefinition<B>,
    P: LayerwisePolicy<B, D::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    proposal_states_with::<B, D, P, _>(
        definition,
        policy,
        embeddings,
        committed,
        end,
        context,
        instrumentation,
        |unit, state, prior, retained, mask, observed| {
            definition.forward_block(unit, state, prior, retained, end, mask, context, observed)
        },
    )
}

/// Runs a TP-local body with the ordinary admission, lease and proposal lifecycle.
/// Inputs are complete replicated embeddings and rank-local committed K/V.
pub fn proposal_states_parallel<B, D, P>(
    definition: &D,
    policy: &mut P,
    embeddings: &B::Tensor,
    committed: &DFlashContext<B::Tensor>,
    end: i32,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend + DistributedNeuralBackend,
    D: TensorParallelAssistantBodyDefinition<B>,
    P: LayerwisePolicy<B, D::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    proposal_states_with::<B, D, P, _>(
        definition,
        policy,
        embeddings,
        committed,
        end,
        context,
        instrumentation,
        |unit, state, prior, retained, mask, observed| {
            definition.forward_block_parallel(
                unit, state, prior, retained, end, mask, parallel, context, observed,
            )
        },
    )
}

fn proposal_states_with<B, D, P, F>(
    definition: &D,
    policy: &mut P,
    embeddings: &B::Tensor,
    committed: &DFlashContext<B::Tensor>,
    end: i32,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    mut block: F,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend,
    D: AssistantBodyDefinition<B>,
    P: LayerwisePolicy<B, D::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    F: FnMut(
        &mut D::Unit,
        &AssistantBodyState<B::Tensor>,
        &DFlashLayerContext<B::Tensor>,
        i32,
        Option<&B::Tensor>,
        &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<AssistantBodyState<B::Tensor>, Error>,
{
    let g = definition.geometry();
    let length = embeddings.shape().get(1).copied().unwrap_or(0);
    let minimum = if g.proposal == ProposalLayout::AnchorThenMasks {
        2
    } else {
        1
    };
    let retained = g.validate_context(committed)?;
    if embeddings.shape() != [1, length, g.hidden]
        || length < minimum
        || length > g.block
        || committed.end != end
    {
        return Err(Error::backend(
            "invalid assistant proposal/context geometry",
        ));
    }
    let (mut state, mask) = definition.begin_proposal(embeddings, committed, end, context)?;
    let mut forward =
        LayerwisePolicyForward::<B, D::Unit, P>::begin(policy, &state.hidden, context)
            .map_err(Error::backend_source)?;
    for (layer, prior) in committed.layers.iter().enumerate() {
        let ordinal = layer + 1;
        let unit = forward
            .acquire(ordinal, g.layout.address(ordinal).unwrap(), |ctx| {
                definition.build_unit(ordinal, ctx)
            })
            .map_err(acquire_error)?;
        let scope = if instrumentation.enabled() {
            format!("layers.{layer}")
        } else {
            String::new()
        };
        state = instrumentation.with_scope(&scope, |observed| {
            block(unit, &state, prior, retained, mask.as_ref(), observed)
        })?;
        forward
            .complete(
                &state.hidden,
                [&prior.keys, &prior.values].into_iter(),
                mask.iter().chain(state.residual.iter()),
            )
            .map_err(Error::backend_source)?;
    }
    let ordinal = g.layers + 1;
    let unit = forward
        .acquire(ordinal, g.layout.address(ordinal).unwrap(), |ctx| {
            definition.build_unit(ordinal, ctx)
        })
        .map_err(acquire_error)?;
    let normalized = definition.normalize(unit, &state, context)?;
    forward
        .complete(
            &normalized,
            state.residual.iter(),
            std::iter::once(&state.hidden),
        )
        .map_err(Error::backend_source)?;
    let output = match g.proposal {
        ProposalLayout::SampleFromAnchor => normalized,
        ProposalLayout::AnchorThenMasks => normalized.index(
            &[Index::Full, Index::Range(1, length), Index::Full],
            context,
        )?,
    };
    forward.finish(&output).map_err(Error::backend_source)?;
    Ok(output)
}
