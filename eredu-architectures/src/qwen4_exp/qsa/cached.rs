//! Cached causal selection over named summary/position streams and fixed tails.
use super::*;
use eredu_core::{
    attention::AttentionPolicy,
    cache::{AppendStreamPolicy, LayerCachePolicy, StateTensorDtype},
};
use eredu_runtime::{AppendStreamSpec, RuntimeAppendStreams, RuntimeStateComponents};

/// Exact mutable ownership for one indexer. Slots are local to the attention unit.
#[derive(Debug, Clone, Copy)]
pub struct QsaStreamStateSpec {
    selection: QsaSelectionSpec,
    partial: QsaPartialStateSpec,
    element: TensorElementType,
    rotary_dimensions: i32,
}
impl QsaStreamStateSpec {
    /// Named complete-summary stream within the owning unit.
    pub const KEYS: u32 = 0;
    /// Named exact original-position stream within the owning unit.
    pub const POSITIONS: u32 = 1;
    /// Creates matching fixed tails and complete-record geometry.
    pub fn new(
        selection: QsaSelectionSpec,
        rotary_dimensions: i32,
        element: TensorElementType,
        rotary_element: TensorElementType,
    ) -> Result<Self, QsaError> {
        selection.validate()?;
        Ok(Self {
            selection,
            partial: QsaPartialStateSpec::new(
                selection.ratio,
                selection.dimensions,
                rotary_dimensions,
                element,
                rotary_element,
                0,
                0,
            )?,
            element,
            rotary_dimensions,
        })
    }
    /// Checks equation/storage agreement before constructing operators.
    pub fn validate_indexer(self, indexer: &QsaIndexerSpec) -> Result<(), QsaError> {
        indexer.validate()?;
        if indexer.selection != self.selection
            || indexer.rotary.dimensions != self.rotary_dimensions
        {
            return Err(QsaError::Geometry);
        }
        Ok(())
    }
    /// Fixed tail tuple, including exact controls and media rotary provenance.
    pub fn partial(self) -> QsaPartialStateSpec {
        self.partial
    }
    /// Two lane-local streams supplied through the shared prepared state contract.
    pub fn streams(self) -> [AppendStreamSpec; 2] {
        [
            AppendStreamSpec {
                slot: Self::KEYS,
                width: self.selection.dimensions,
                element: self.element,
            },
            AppendStreamSpec {
                slot: Self::POSITIONS,
                width: self.selection.ratio,
                element: TensorElementType::I32,
            },
        ]
    }
    /// Ordinary K/V history, bounded fixed tails and sealable summary records.
    /// Sparse selection does not reduce the retained ordinary K/V prefix.
    pub fn cache_policy(self, kv_heads: i32, head_dim: i32) -> Result<LayerCachePolicy, QsaError> {
        let dtype = if self.element == TensorElementType::F32 {
            StateTensorDtype::Float32
        } else {
            StateTensorDtype::Floating
        };
        let streams = vec![
            AppendStreamPolicy::new(
                Self::KEYS,
                self.selection.dimensions,
                dtype,
                self.selection.ratio,
            )
            .map_err(Error::backend)?,
            AppendStreamPolicy::new(
                Self::POSITIONS,
                self.selection.ratio,
                StateTensorDtype::Int32,
                self.selection.ratio,
            )
            .map_err(Error::backend)?,
        ];
        Ok(LayerCachePolicy::key_value_with_state(
            AttentionPolicy::Full,
            kv_heads,
            head_dim,
            self.partial.policies(),
            streams,
        )
        .map_err(Error::backend)?)
    }
}

/// Finite invocation bounds, separate from append storage and ordinary K/V budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaExecutionLimits {
    /// Maximum batch lanes admitted by this executable.
    pub batch: i32,
    /// Maximum tokens per lane in one invocation.
    pub tokens: i32,
    /// Architecture-owned tensor/host workspace, including the selected-position result.
    /// Backend projection/operator workspace is admitted separately by its mechanisms.
    pub workspace_bytes: u64,
}
impl QsaExecutionLimits {
    /// Conservative maximum for projections, selected positions, tail packing and
    /// one bounded selection tile. No sequence-by-sequence object is included.
    pub fn required_workspace<T: Tensor>(self, state: QsaStreamStateSpec) -> Result<u64, QsaError> {
        if self.batch <= 0 || self.tokens <= 0 {
            return Err(QsaError::Geometry);
        }
        let s = state.selection;
        let rows = (self.batch as u64)
            .checked_mul(self.tokens as u64)
            .ok_or(QsaError::Geometry)?;
        let projected = (s.heads as u64 + 1)
            .checked_mul(s.dimensions as u64)
            .and_then(|n| n.checked_mul(rows))
            .and_then(|n| n.checked_mul(32))
            .ok_or(QsaError::Geometry)?;
        let selected = (s.token_budget as u64 + s.ratio as u64 - 1)
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(16))
            .ok_or(QsaError::Geometry)?;
        projected
            .checked_add(selected)
            .and_then(|n| {
                n.checked_add(
                    state
                        .partial
                        .required_workspace::<T>(self.batch as usize)
                        .ok()?,
                )
            })
            .and_then(|n| n.checked_add(s.required_workspace().ok()?))
            .ok_or(QsaError::Geometry)
    }
    pub(crate) fn validate<T: Tensor>(self, state: QsaStreamStateSpec) -> Result<(), QsaError> {
        let required = self.required_workspace::<T>(state)?;
        if required > self.workspace_bytes {
            return Err(QsaError::Workspace {
                required,
                limit: self.workspace_bytes,
            });
        }
        Ok(())
    }
}

/// One indexer invocation over causal visibility and current-token rotary values.
pub struct QsaSelectionInput<'a, T> {
    /// Collapsed residual input `[batch,tokens,hidden]`.
    pub hidden: &'a T,
    /// Exact current-prefix position, before ordinary attention appends K/V.
    pub offset: i32,
    /// Optional row-major padding visibility; no dense attention mask is needed.
    pub visible: Option<&'a [bool]>,
    /// Current text positions or media products `[batch or 1,tokens,rotary_dim]`.
    /// The equivalent `[batch or 1,1,tokens,rotary_dim]` form is also accepted.
    pub rotary: RotaryPosition<'a, T>,
}

/// Architecture coordinator; all mutable values live in the shared runtime state.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CachedQsaSelector<B: NeuralBackend> {
    indexer: QsaIndexer<B>,
    #[parameter(skip)]
    state: QsaStreamStateSpec,
    #[parameter(skip)]
    limits: QsaExecutionLimits,
}
impl<B: NeuralBackend> CachedQsaSelector<B> {
    /// Rejects mismatched equations/storage and unadmitted scratch before allocation.
    pub fn new(
        indexer: QsaIndexerSpec,
        state: QsaStreamStateSpec,
        limits: QsaExecutionLimits,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, QsaError> {
        limits.validate::<B::Tensor>(state)?;
        state.validate_indexer(&indexer)?;
        B::require_operator_capabilities(
            "cached QSA",
            C::FROM_I32_SLICE.union(C::CAST_FLOAT).union(C::TO_I32_VEC),
        )?;
        Ok(Self {
            indexer: QsaIndexer::new(indexer, context)?,
            state,
            limits,
        })
    }
    /// Publishes summaries and tails without advancing the ordinary K/V prefix.
    /// The surrounding attention-unit transaction checkpoints all three kinds of
    /// state and restores them together on failure, cancellation or rejected drafts.
    pub fn select<S>(
        &mut self,
        input: QsaSelectionInput<'_, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, QsaError>
    where
        S: RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        let shape = input.hidden.shape();
        if shape.len() != 3
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[0] > self.limits.batch
            || shape[1] > self.limits.tokens
            || input.offset < 0
            || input.offset.checked_add(shape[1]).is_none()
            || state.position() != input.offset
        {
            return Err(QsaError::Geometry);
        }
        self.validate_rotary(input.rotary, shape[0], shape[1])?;
        let (batch, tokens) = (shape[0] as usize, shape[1] as usize);
        let rows = batch.checked_mul(tokens).ok_or(QsaError::Geometry)?;
        if input.visible.is_some_and(|v| v.len() != rows) {
            return Err(QsaError::Geometry);
        }
        let scratch = self.state.partial.required_workspace::<B::Tensor>(batch)?;
        let mut partials = self
            .state
            .partial
            .read::<B, _>(state, batch, scratch, context)?;
        let specs = self.state.streams();
        // Validate every lane before submitting projection work or appending records.
        for (lane, partial) in partials.iter().enumerate() {
            if partial.last_seen() != input.offset.checked_sub(1).filter(|v| *v >= 0) {
                return Err(QsaError::Geometry);
            }
            let (keys, positions) = state.append_stream_pair(
                QsaStreamStateSpec::KEYS,
                QsaStreamStateSpec::POSITIONS,
                lane as u32,
            )?;
            if keys.specification() != &specs[0]
                || positions.specification() != &specs[1]
                || keys.len() != positions.len()
                || keys
                    .len()
                    .checked_mul(self.state.selection.ratio as usize)
                    .and_then(|n| n.checked_add(partial.tail().len()))
                    .is_none_or(|n| n > input.offset as usize)
            {
                return Err(QsaError::Geometry);
            }
        }
        let (queries, raw) = self.indexer.project(input.hidden, input.rotary, context)?;
        if raw.element_type() != Some(self.state.element) {
            return Err(QsaError::Geometry);
        }
        let slots = self.state.selection.token_budget + self.state.selection.ratio - 1;
        let mut selected = Vec::with_capacity(rows);
        for (lane, partial) in partials.iter_mut().enumerate() {
            let (keys, positions) = state.append_stream_pair(
                QsaStreamStateSpec::KEYS,
                QsaStreamStateSpec::POSITIONS,
                lane as u32,
            )?;
            for token in 0..tokens {
                let position = input.offset + token as i32;
                let key = raw
                    .index(
                        &[
                            Index::Range(lane as i32, lane as i32 + 1),
                            Index::Range(token as i32, token as i32 + 1),
                            Index::Full,
                        ],
                        context,
                    )?
                    .reshape(&[1, self.state.selection.dimensions], context)?;
                let rotary = token_rotary(input.rotary, lane as i32, token as i32, context)?;
                let visible = input.visible.is_none_or(|v| v[lane * tokens + token]);
                if let Some(block) =
                    partial.push(&key, position, visible, rotary.as_position(), context)?
                {
                    let summary = self.indexer.summarize(
                        &block.raw_keys,
                        block.first_position.as_position(),
                        context,
                    )?;
                    let values = B::Tensor::from_i32_slice(
                        &block.positions,
                        &[1, self.state.selection.ratio],
                        context,
                    )?;
                    let frontier = keys.len();
                    keys.append(frontier, summary, context)?;
                    positions.append(frontier, values, context)?;
                }
                let query = queries
                    .index(
                        &[
                            Index::Range(lane as i32, lane as i32 + 1),
                            Index::Full,
                            Index::Range(token as i32, token as i32 + 1),
                            Index::Full,
                        ],
                        context,
                    )?
                    .reshape(
                        &[self.state.selection.heads, self.state.selection.dimensions],
                        context,
                    )?;
                // Prefill completes each query's selector before retaining the
                // next; otherwise lazy score tiles would accumulate per token.
                let native = if tokens == 1 {
                    select_positions_tensor(
                        self.state.selection,
                        &query,
                        keys,
                        positions,
                        partial.tail(),
                        position + 1,
                        context,
                    )?
                } else {
                    None
                };
                let values = match native {
                    Some(values) => values,
                    None => {
                        let values = select_positions(
                            self.state.selection,
                            &query,
                            keys,
                            positions,
                            partial.tail(),
                            context,
                        )?;
                        if values.iter().any(|v| *v > position) {
                            return Err(QsaError::Geometry);
                        }
                        B::Tensor::from_i32_slice(&values, &[slots], context)?
                    }
                };
                selected.push(values);
            }
        }
        let selected = B::Tensor::concatenate(&selected, 0, context)?
            .reshape(&[shape[0], shape[1], slots], context)?;
        self.state
            .partial
            .write::<B, _>(state, &partials, scratch, context)?;
        Ok(selected)
    }
    fn validate_rotary(
        &self,
        position: RotaryPosition<'_, B::Tensor>,
        batch: i32,
        tokens: i32,
    ) -> Result<(), QsaError> {
        match position {
            RotaryPosition::Offset(offset)
                if offset >= 0 && offset.checked_add(tokens).is_some() =>
            {
                Ok(())
            }
            RotaryPosition::Embeddings { cosine, sine } if cosine.shape() == sine.shape() => {
                let valid = match cosine.shape() {
                    [b, t, d] => {
                        (*b == 1 || *b == batch)
                            && *t == tokens
                            && *d == self.state.rotary_dimensions
                    }
                    [b, 1, t, d] => {
                        (*b == 1 || *b == batch)
                            && *t == tokens
                            && *d == self.state.rotary_dimensions
                    }
                    _ => false,
                };
                if valid {
                    Ok(())
                } else {
                    Err(QsaError::Geometry)
                }
            }
            _ => Err(QsaError::Geometry),
        }
    }
}
fn token_rotary<T: Tensor>(
    position: RotaryPosition<'_, T>,
    lane: i32,
    token: i32,
    context: &T::Context,
) -> Result<QsaBlockPosition<T>, QsaError> {
    Ok(match position {
        RotaryPosition::Offset(offset) => QsaBlockPosition::Offset(offset + token),
        RotaryPosition::Embeddings { cosine, sine } => {
            let b = if cosine.dim(0) == 1 { 0 } else { lane };
            let slice = |value: &T| {
                let mut axes = vec![Index::Range(b, b + 1)];
                if value.shape().len() == 4 {
                    axes.push(Index::Full);
                }
                axes.extend([Index::Range(token, token + 1), Index::Full]);
                value.index(&axes, context)
            };
            QsaBlockPosition::Embeddings {
                cosine: slice(cosine)?,
                sine: slice(sine)?,
            }
        }
    })
}
