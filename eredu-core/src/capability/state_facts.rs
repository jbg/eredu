//! The existing logical-state equations with borrowed diagnostic destinations.
//! These declarations never prove native backing/workspace or grant execution.
use super::*;

pub(super) fn checked_add(
    a: u64,
    b: u64,
    operation: &'static str,
) -> Result<u64, AdmissionPolicyError> {
    a.checked_add(b)
        .ok_or(AdmissionPolicyError::ArithmeticOverflow { operation })
}
fn checked_mul(a: u64, b: u64, operation: &'static str) -> Result<u64, AdmissionPolicyError> {
    a.checked_mul(b)
        .ok_or(AdmissionPolicyError::ArithmeticOverflow { operation })
}

/// Scalar projection of the same immutable executable state layout.
/// Selected native backing and execution workspace remain separate requirements.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeStateFacts<'a> {
    /// Context-independent recurrent/convolution state.
    pub fixed_state_bytes: u64,
    /// Logical unbounded growth before multiplying by batch.
    pub bytes_per_position_per_batch: u64,
    /// Context-dependent state under the exact layer policies.
    pub context_state_bytes: u64,
    /// Legacy input-declared retained media embedding term.
    pub multimodal_embedding_bytes: u64,
    /// Legacy input-declared media workspace; not native primitive coverage.
    pub media_execution_workspace_bytes: u64,
    /// The existing checked logical sum, without selected native capacity.
    pub requested_state_bytes: u64,
    /// Coverage of logical persistent state, not execution.
    pub persistent_state_completeness: EstimationCompleteness,
    /// Overall coverage remains persistent-state-only.
    pub completeness: EstimationCompleteness,
    /// Selected physical width for generic floating state.
    pub floating_state_dtype_bytes: NonZeroU8,
    /// Validated batch extent.
    pub batch_size: u64,
    /// Prompt plus generated allowance.
    pub requested_positions: u64,
    /// Original full-cache growth granularity.
    pub allocation_granularity: u64,
    /// Immutable sorted distinct-window projection from the same layout.
    pub sliding_windows: StateWindowPlan<'a>,
}
impl RuntimeStateFacts<'_> {
    pub(super) fn into_estimate(self) -> RuntimeStateEstimate {
        let windows = self.sliding_windows.iter().collect();
        self.build_estimate(windows)
    }

    /// Consumes an independently prepared exact diagnostic destination. The
    /// same borrowed window plan validates values before ownership is installed.
    pub fn into_estimate_with_windows(
        self,
        windows: Vec<u64>,
    ) -> Result<RuntimeStateEstimate, AdmissionPolicyError> {
        if !windows.iter().copied().eq(self.sliding_windows.iter()) {
            return Err(AdmissionPolicyError::InvalidConfiguration {
                field: "state_windows",
                detail: "state window destination differs from the source layout",
            });
        }
        Ok(self.build_estimate(windows))
    }

    fn build_estimate(self, windows: Vec<u64>) -> RuntimeStateEstimate {
        RuntimeStateEstimate {
            fixed_state_bytes: self.fixed_state_bytes,
            bytes_per_position_per_batch: self.bytes_per_position_per_batch,
            context_state_bytes: self.context_state_bytes,
            selected_state_backing: None,
            multimodal_embedding_bytes: self.multimodal_embedding_bytes,
            media_execution_workspace_bytes: self.media_execution_workspace_bytes,
            requested_state_bytes: self.requested_state_bytes,
            execution_workspace: None,
            persistent_state_completeness: self.persistent_state_completeness,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: self.floating_state_dtype_bytes,
                batch_size: self.batch_size,
                requested_positions: self.requested_positions,
                sliding_window_bounds: windows,
                allocation_granularity: self.allocation_granularity,
            },
            completeness: self.completeness,
        }
    }
}

/// Fixed destination refusal. No element is written unless the exact length fits.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("state window destination has {actual} entries; expected {expected}")]
pub struct StateWindowDestinationError {
    /// Exact distinct source-window count.
    pub expected: usize,
    /// Supplied destination length.
    pub actual: usize,
}

/// Borrowed canonical windows of one immutable validated state schedule.
/// Construction and iteration allocate no sorting or shape storage.
#[derive(Debug, Clone, Copy)]
pub struct StateWindowPlan<'a> {
    layout: &'a StateMemoryLayout,
    count: usize,
}
fn next_window(layout: &StateMemoryLayout, after: Option<u64>) -> Option<u64> {
    layout
        .layer_layout()
        .iter()
        .filter_map(|policy| match policy.attention() {
            Some(AttentionPolicy::Sliding { window }) => Some(u64::from(window.get())),
            _ => None,
        })
        .filter(|value| after.is_none_or(|after| *value > after))
        .min()
}
impl<'a> StateWindowPlan<'a> {
    fn new(layout: &'a StateMemoryLayout) -> Result<Self, AdmissionPolicyError> {
        let mut after = None;
        let mut count = 0usize;
        while let Some(value) = next_window(layout, after) {
            count = count
                .checked_add(1)
                .ok_or(AdmissionPolicyError::ArithmeticOverflow {
                    operation: "distinct state window count",
                })?;
            after = Some(value);
        }
        Ok(Self { layout, count })
    }
    /// Exact number of distinct positive windows.
    pub const fn len(&self) -> usize {
        self.count
    }
    /// Whether this schedule has no sliding attention layer.
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Ascending distinct windows, using repeated borrowed scans.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = u64> + Clone + 'a {
        WindowIter {
            layout: self.layout,
            after: None,
            remaining: self.count,
        }
    }
    /// Fill only an exact destination. Refusal never writes a partial prefix.
    pub fn fill(&self, destination: &mut [u64]) -> Result<(), StateWindowDestinationError> {
        if destination.len() != self.count {
            return Err(StateWindowDestinationError {
                expected: self.count,
                actual: destination.len(),
            });
        }
        for (slot, window) in destination.iter_mut().zip(self.iter()) {
            *slot = window;
        }
        Ok(())
    }
}
#[derive(Clone)]
struct WindowIter<'a> {
    layout: &'a StateMemoryLayout,
    after: Option<u64>,
    remaining: usize,
}
impl Iterator for WindowIter<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        let value = next_window(self.layout, self.after)?;
        self.after = Some(value);
        self.remaining -= 1;
        Some(value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for WindowIter<'_> {}

fn attention_scalars_per_position(policy: &LayerCachePolicy) -> Result<u64, AdmissionPolicyError> {
    let scalars = match policy {
        LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyValueWithFixedState {
            num_key_value_heads,
            head_dim,
            ..
        } => checked_mul(
            checked_mul(
                u64::from(num_key_value_heads.get()),
                u64::from(head_dim.get()),
                "key/value heads times head dimension",
            )?,
            2,
            "key plus value scalars",
        )?,
        LayerCachePolicy::KeyOnly {
            num_key_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyOnlyWithFixedState {
            num_key_heads,
            head_dim,
            ..
        } => checked_mul(
            u64::from(num_key_heads.get()),
            u64::from(head_dim.get()),
            "key heads times head dimension",
        )?,
        LayerCachePolicy::CompressedLatentRotary {
            latent_dim,
            rotary_dim,
            ..
        } => checked_add(
            u64::from(latent_dim.get()),
            u64::from(rotary_dim.get()),
            "compressed latent plus rotary width",
        )?,
        LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => 0,
    };
    Ok(scalars)
}

fn is_context_dependent_dimension(dimension: &StateTensorDimension) -> bool {
    matches!(
        dimension,
        StateTensorDimension::PrefixTokens
            | StateTensorDimension::PrefixTokensDiv(_)
            | StateTensorDimension::PrefixTokensRem(_)
    )
}

fn state_tensor_dtype_bytes(tensor: &StateTensorPolicy, floating_scalar_bytes: u64) -> u64 {
    match tensor.dtype {
        StateTensorDtype::Floating => floating_scalar_bytes,
        StateTensorDtype::Float32 | StateTensorDtype::Int32 | StateTensorDtype::Uint32 => 4,
    }
}

fn state_tensor_is_present(tensor: &StateTensorPolicy, prefix_tokens: usize) -> bool {
    match tensor.presence {
        StateTensorPresence::Required => true,
        // Prepared prefix embeddings are accounted once through the input's
        // authoritative media-position count below. Any other optional state
        // is included conservatively because its request-time presence is not
        // otherwise represented in the portable input descriptor.
        StateTensorPresence::Optional => !matches!(tensor.role, StateTensorRole::PrefixEmbedding),
        StateTensorPresence::PrefixRemainderNonZero(divisor) => {
            !prefix_tokens.is_multiple_of(divisor.get() as usize)
        }
        StateTensorPresence::PrefixAtLeast(divisor) => prefix_tokens >= divisor.get() as usize,
    }
}

fn state_tensor_bytes(
    tensor: &StateTensorPolicy,
    batch_size: usize,
    prefix_tokens: usize,
    floating_scalar_bytes: u64,
) -> Result<u64, AdmissionPolicyError> {
    if !state_tensor_is_present(tensor, prefix_tokens) {
        return Ok(0);
    }
    // Legacy resolved_shape validates every dimension before scalar products.
    // Both scans borrow the same immutable policy, so late conversion failure
    // still wins over an earlier possible product overflow or zero extent.
    let dimensions = || {
        tensor
            .resolved_dimensions(batch_size, prefix_tokens)
            .map(|dimension| {
                dimension.map_err(|error| AdmissionPolicyError::InvalidConfiguration {
                    field: "layer_layout",
                    detail: error.diagnostic(),
                })
            })
    };
    dimensions().try_for_each(|dimension| dimension.map(|_| ()))?;
    let scalars = dimensions().try_fold(1_u64, |scalars, dimension| {
        checked_mul(
            scalars,
            u64::try_from(dimension?).map_err(|_| AdmissionPolicyError::InvalidConfiguration {
                field: "layer_layout",
                detail: "runtime state tensor has a negative resolved dimension",
            })?,
            "runtime state tensor scalar count",
        )
    })?;
    checked_mul(
        scalars,
        state_tensor_dtype_bytes(tensor, floating_scalar_bytes),
        "runtime state tensor bytes",
    )
}

fn state_tensor_bytes_per_position_per_batch(
    tensor: &StateTensorPolicy,
    floating_scalar_bytes: u64,
) -> Result<u64, AdmissionPolicyError> {
    let mut scalars = 1_u64;
    let mut divisor = 1_u64;
    let mut unbounded = false;
    for dimension in &tensor.shape {
        match dimension {
            StateTensorDimension::Batch | StateTensorDimension::Scalar => {}
            StateTensorDimension::Fixed(value) => {
                scalars =
                    checked_mul(scalars, u64::from(value.get()), "state growth scalar count")?;
            }
            StateTensorDimension::PrefixTokens => unbounded = true,
            StateTensorDimension::PrefixTokensDiv(value) => {
                unbounded = true;
                divisor = checked_mul(divisor, u64::from(value.get()), "state growth divisor")?;
            }
            StateTensorDimension::PrefixTokensRem(_) => return Ok(0),
        }
    }
    if !unbounded {
        return Ok(0);
    }
    let bytes = checked_mul(
        scalars,
        state_tensor_dtype_bytes(tensor, floating_scalar_bytes),
        "state growth bytes",
    )?;
    Ok(bytes.div_ceil(divisor))
}

/// Computes the ordinary logical-state equations without a shape/window buffer.
/// A complete native request still needs selected backing, primitive workspace,
/// source custody and the existing single admission comparison.
pub fn estimate_runtime_state_facts<'a>(
    layout: &'a StateMemoryLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    floating_state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateFacts<'a>, AdmissionPolicyError> {
    if batch_size == 0 {
        return Err(AdmissionPolicyError::InvalidConfiguration {
            field: "batch_size",
            detail: "must be positive".into(),
        });
    }
    let requested_positions = checked_add(
        input.model_positions,
        max_output_tokens,
        "prompt plus output positions",
    )?;
    let floating_scalar_bytes = u64::from(floating_state_dtype_bytes.get());
    let batch_size_usize =
        usize::try_from(batch_size).map_err(|_| AdmissionPolicyError::InvalidConfiguration {
            field: "batch_size",
            detail: "exceeds the runtime state shape range".into(),
        })?;
    let mut fixed_state_bytes = 0;
    let mut context_state_bytes = 0;
    let mut unbounded_per_position = 0;
    for (layer, policy) in layout.layer_layout.iter().enumerate() {
        let layer_positions = requested_positions
            .saturating_sub(u64::from(layout.layer_prefix_offsets[layer].unsigned_abs()));
        let layer_positions_usize = usize::try_from(layer_positions).map_err(|_| {
            AdmissionPolicyError::InvalidConfiguration {
                field: "requested_positions",
                detail: "exceeds the runtime state shape range".into(),
            }
        })?;
        if let Some(attention) = policy.attention() {
            let per_position = attention_scalars_per_position(policy)?;
            let retained = match attention {
                AttentionPolicy::Sliding { window } => {
                    let window = u64::from(window.get());
                    layer_positions.min(window)
                }
                AttentionPolicy::Full => {
                    let adjustment = layout.allocation_granularity - 1;
                    checked_add(layer_positions, adjustment, "cache allocation rounding")?
                        / layout.allocation_granularity
                        * layout.allocation_granularity
                }
            };
            let bytes = checked_mul(
                checked_mul(
                    checked_mul(per_position, retained, "attention context scalars")?,
                    batch_size,
                    "attention context batch",
                )?,
                floating_scalar_bytes,
                "attention context bytes",
            )?;
            context_state_bytes =
                checked_add(context_state_bytes, bytes, "context state byte total")?;
            if matches!(attention, AttentionPolicy::Full) {
                unbounded_per_position = checked_add(
                    unbounded_per_position,
                    checked_mul(
                        per_position,
                        floating_scalar_bytes,
                        "unbounded bytes per position",
                    )?,
                    "unbounded bytes-per-position total",
                )?;
            }
        }
        for tensor in policy.fixed_state() {
            let bytes = state_tensor_bytes(
                tensor,
                batch_size_usize,
                layer_positions_usize,
                floating_scalar_bytes,
            )?;
            if tensor.shape.iter().any(is_context_dependent_dimension) {
                context_state_bytes =
                    checked_add(context_state_bytes, bytes, "context state byte total")?;
                unbounded_per_position = checked_add(
                    unbounded_per_position,
                    state_tensor_bytes_per_position_per_batch(tensor, floating_scalar_bytes)?,
                    "unbounded bytes-per-position total",
                )?;
            } else {
                fixed_state_bytes =
                    checked_add(fixed_state_bytes, bytes, "fixed state byte total")?;
            }
        }
    }
    let sliding_windows = StateWindowPlan::new(layout)?;
    let multimodal_embedding_bytes = checked_mul(
        checked_mul(
            checked_mul(
                input.media_positions,
                layout.hidden_size,
                "media positions times hidden size",
            )?,
            batch_size,
            "media embeddings times batch",
        )?,
        floating_scalar_bytes,
        "media embedding bytes",
    )?;
    let media_execution_workspace_bytes = checked_mul(
        input.media_execution_workspace_bytes,
        batch_size,
        "media execution workspace times batch",
    )?;
    let requested_state_bytes = checked_add(
        checked_add(
            checked_add(
                fixed_state_bytes,
                context_state_bytes,
                "fixed plus context state",
            )?,
            multimodal_embedding_bytes,
            "persistent plus multimodal embedding state",
        )?,
        media_execution_workspace_bytes,
        "persistent plus media execution workspace",
    )?;
    // A state layout never bounds native text execution workspace. In
    // particular, a conservative media estimate cannot upgrade missing text
    // bounds to a complete estimate.
    let completeness = EstimationCompleteness::PersistentStateOnly;
    Ok(RuntimeStateFacts {
        fixed_state_bytes,
        bytes_per_position_per_batch: unbounded_per_position,
        context_state_bytes,
        multimodal_embedding_bytes,
        media_execution_workspace_bytes,
        requested_state_bytes,
        persistent_state_completeness: if layout.completeness
            == EstimationCompleteness::PersistentStateOnly
        {
            layout.completeness
        } else {
            match (input.kind, input.media_execution_workspace_kind) {
                (ObservationKind::Exact, ObservationKind::Exact) => layout.completeness,
                (
                    ObservationKind::Exact | ObservationKind::Conservative,
                    ObservationKind::Exact | ObservationKind::Conservative,
                ) => EstimationCompleteness::Conservative,
                _ => EstimationCompleteness::PersistentStateOnly,
            }
        },
        floating_state_dtype_bytes,
        batch_size,
        requested_positions,
        sliding_windows,
        allocation_granularity: layout.allocation_granularity,
        completeness,
    })
}

#[cfg(test)]
mod tests;
