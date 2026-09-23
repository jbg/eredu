use super::*;
use crate::{
    AttentionCache, AttentionRequest, CausalDepthwiseConvolutionSpec, GatedDeltaScanInput,
    GroupedLinearSpec, LinearSpec, SelectiveStateSpaceScanInput, Tensor,
};

fn positive(value: i32) -> Result<u64, Error> {
    u64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| Error::backend("mechanism dimension must be positive"))
}
fn shape4(shape: &[i32]) -> Result<[u64; 4], Error> {
    if shape.len() != 4 {
        return Err(Error::backend("mechanism expects rank-four input"));
    }
    Ok([
        positive(shape[0])?,
        positive(shape[1])?,
        positive(shape[2])?,
        positive(shape[3])?,
    ])
}
fn checked(invocation: MechanismInvocation) -> Result<MechanismInvocation, Error> {
    invocation.logical_values()?;
    Ok(invocation)
}

impl LinearSpec {
    /// Describes a selected projection without constructing a native operator.
    pub fn memory_invocation(
        &self,
        rows: u64,
        element: TensorElementType,
        weight_element: Option<TensorElementType>,
    ) -> Result<MechanismInvocation, Error> {
        self.format.validate_for_weight(&self.weight)?;
        checked(MechanismInvocation::Projection {
            rows,
            input: positive(self.input)?,
            output: positive(self.output)?,
            format: self.format.encoding(),
            element,
            weight_element,
            bias: self.bias.is_some(),
        })
    }
}

impl<T: Tensor> AttentionRequest<'_, T> {
    /// Describes actual request geometry by reading shape metadata only.
    pub fn memory_invocation(
        &self,
        element: TensorElementType,
    ) -> Result<MechanismInvocation, Error> {
        self.validate()?;
        let [batch, query_heads, queries, key_width] = shape4(self.queries.shape())?;
        let [_, kv_heads, keys, _] = shape4(self.keys.shape())?;
        let [_, _, _, value_width] = shape4(self.values.shape())?;
        checked(MechanismInvocation::Attention {
            batch,
            query_heads,
            kv_heads,
            queries,
            keys,
            key_width,
            value_width,
            element,
            arithmetic: self.arithmetic,
            softcap: self.softcap.is_some(),
            sinks: self.sinks.is_some(),
        })
    }
}

impl CausalDepthwiseConvolutionSpec {
    /// Describes output and bounded history from the ordinary convolution spec.
    pub fn memory_invocation(
        &self,
        batch: u64,
        tokens: u64,
        element: TensorElementType,
    ) -> Result<MechanismInvocation, Error> {
        self.validate()?;
        checked(MechanismInvocation::Convolution {
            batch,
            tokens,
            channels: positive(self.channels)?,
            kernel: positive(self.kernel_size)?,
            element,
        })
    }
}

impl GroupedLinearSpec {
    /// Describes selected expert projections using local bank/output geometry.
    pub fn memory_invocation(
        &self,
        rows: u64,
        selected: u64,
        element: TensorElementType,
    ) -> Result<MechanismInvocation, Error> {
        self.validate()?;
        checked(MechanismInvocation::ExpertDispatch {
            rows,
            experts: positive(self.group_count())?,
            selected,
            input: positive(self.input_dimensions())?,
            output: positive(self.output_dimensions())?,
            format: self.projection().format().encoding(),
            element,
        })
    }
}

impl<T: Tensor> GatedDeltaScanInput<'_, T> {
    /// Describes matrix-state and output geometry without running the recurrence.
    pub fn memory_invocation(
        &self,
        element: TensorElementType,
    ) -> Result<MechanismInvocation, Error> {
        let [batch, tokens, heads, state_width] = shape4(self.query.shape())?;
        let [value_batch, value_tokens, value_heads, value_width] = shape4(self.value.shape())?;
        if self.key.shape() != self.query.shape()
            || [batch, tokens, heads] != [value_batch, value_tokens, value_heads]
        {
            return Err(Error::backend("recurrent query/key/value geometry differs"));
        }
        let decay = self.log_decay.shape();
        let base = [batch as i32, tokens as i32, heads as i32];
        if self.beta.shape() != base || (decay != base && decay != self.query.shape()) {
            return Err(Error::backend("recurrent decay/beta geometry differs"));
        }
        if self.initial_state.is_some_and(|state| {
            state.shape()
                != [
                    batch as i32,
                    heads as i32,
                    state_width as i32,
                    value_width as i32,
                ]
        }) {
            return Err(Error::backend("gated-delta initial state geometry differs"));
        }
        checked(MechanismInvocation::Recurrent {
            kind: RecurrentKind::GatedDelta,
            batch,
            tokens,
            heads,
            value_width,
            state_width,
            element,
            chunk_size: None,
        })
    }
}

impl<T: Tensor> SelectiveStateSpaceScanInput<'_, T> {
    /// Describes selected scan geometry and its explicit chunk policy.
    pub fn memory_invocation(
        &self,
        element: TensorElementType,
    ) -> Result<MechanismInvocation, Error> {
        let [batch, tokens, heads, value_width] = shape4(self.values.shape())?;
        let [b, t, h, state_width] = shape4(self.input_state.shape())?;
        if [batch, tokens, heads] != [b, t, h]
            || self.output_state.shape() != self.input_state.shape()
            || self.time_step.shape() != [batch as i32, tokens as i32, heads as i32]
            || [self.time_step_bias, self.transition_log, self.skip]
                .iter()
                .any(|value| value.shape() != [heads as i32])
            || self.initial_state.is_some_and(|state| {
                state.shape()
                    != [
                        batch as i32,
                        heads as i32,
                        value_width as i32,
                        state_width as i32,
                    ]
            })
        {
            return Err(Error::backend(
                "selective state-space input geometry differs",
            ));
        }
        checked(MechanismInvocation::Recurrent {
            kind: RecurrentKind::SelectiveStateSpace,
            batch,
            tokens,
            heads,
            value_width,
            state_width,
            element,
            chunk_size: Some(self.chunk_size as u64),
        })
    }
}

/// Derives an append's logical inputs and visible history from actual cache metadata.
/// The cache's implementation hook determines backing reuse, compression and spare capacity.
/// The storage hook must provide retained length: an absolute sliding-cache
/// offset cannot substitute for it. Missing metadata returns an explicit error.
pub fn cache_update_invocation<T: Tensor, C: AttentionCache<T> + ?Sized>(
    cache: &C,
    keys: &T,
    values: &T,
    element: TensorElementType,
) -> Result<MechanismInvocation, Error> {
    let retained_positions = cache
        .memory_retained_positions()
        .ok_or_else(|| Error::backend("cache retained memory geometry is unavailable"))?;
    let [batch, heads, appended, key_width] = shape4(keys.shape())?;
    let [b, h, t, value_width] = shape4(values.shape())?;
    if [batch, heads, appended] != [b, h, t] {
        return Err(Error::backend("cache key/value append geometry differs"));
    }
    checked(MechanismInvocation::CacheUpdate {
        batch,
        heads,
        previous: retained_positions,
        appended,
        key_width,
        value_width,
        element,
    })
}
