//! QSA micro-block summaries and bounded selection over original token positions.
use eredu_nn::{
    Error, Index, LinearOperator, LinearSpec, NeuralBackend, NeuralOperatorCapabilities as C,
    NormalizationConstructionSpec, NormalizationOperator, RotaryOperator, RotaryPosition,
    RotarySpec, Tensor, TensorElementType,
};
use eredu_runtime::{AppendOnlyStream, AppendStreamError};
use std::{cmp::Ordering, collections::BinaryHeap};
mod cached;
mod state;
pub use cached::{CachedQsaSelector, QsaExecutionLimits, QsaSelectionInput, QsaStreamStateSpec};
pub use state::QsaPartialStateSpec;

/// Architecture geometry and selected execution workspace bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaSelectionSpec {
    /// Query index heads.
    pub heads: i32,
    /// Index feature width.
    pub dimensions: i32,
    /// Visible original tokens per complete micro-block.
    pub ratio: i32,
    /// Maximum complete-block tokens selected for one query.
    pub token_budget: i32,
    /// Maximum summaries read and scored together.
    pub tile_blocks: i32,
    /// Admitted transient tensor/host workspace, separate from retained history.
    pub workspace_bytes: u64,
}
/// Invalid equation geometry, state, selection workspace or backend failure.
#[derive(Debug, thiserror::Error)]
pub enum QsaError {
    /// Declared state binding or publication failed.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),
    /// Invalid geometry or malformed retained state.
    #[error("invalid QSA geometry, original positions or summary state")]
    Geometry,
    /// Execution scratch was not admitted.
    #[error("QSA workspace requires {required} bytes, admitted {limit}")]
    Workspace {
        /// Required bound.
        required: u64,
        /// Admitted bound.
        limit: u64,
    },
    /// A score is not a finite real value.
    #[error("QSA index score is not finite")]
    Score,
    /// Bounded stream read or append failure.
    #[error(transparent)]
    Stream(#[from] AppendStreamError),
    /// Generic tensor construction or execution failure.
    #[error(transparent)]
    Tensor(#[from] Error),
}
impl QsaSelectionSpec {
    /// Conservative logical workspace for a single query. Retained K/V and summary
    /// streams are accounted separately and continue to grow with sequence length.
    pub fn required_workspace(&self) -> Result<u64, QsaError> {
        if self.heads <= 0
            || self.dimensions <= 0
            || self.ratio <= 0
            || self.token_budget <= 0
            || self.token_budget % self.ratio != 0
            || self.tile_blocks <= 0
            || self.token_budget.checked_add(self.ratio - 1).is_none()
        {
            return Err(QsaError::Geometry);
        }
        let h = self.heads as u64;
        let d = self.dimensions as u64;
        let tile = self.tile_blocks as u64;
        let slots = (self.token_budget + self.ratio - 1) as u64;
        // Input/FP32 compact tiles, score/masked/reduced arrays, compact query,
        // bounded host scores/top-k records/positions and selected output storage.
        let tensor = tile
            .checked_mul(d)
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| {
                tile.checked_mul(h)
                    .and_then(|v| v.checked_mul(16))
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| {
                h.checked_mul(d)
                    .and_then(|v| v.checked_mul(8))
                    .and_then(|v| n.checked_add(v))
            })
            .ok_or(QsaError::Geometry)?;
        tensor
            .checked_add(
                tile.checked_mul(self.ratio as u64)
                    .and_then(|n| n.checked_mul(16))
                    .ok_or(QsaError::Geometry)?,
            )
            .and_then(|n| n.checked_add(tile.checked_mul(16)?))
            .and_then(|n| slots.checked_mul(64).and_then(|v| n.checked_add(v)))
            .ok_or(QsaError::Geometry)
    }
    /// Checks scratch before any stream acquisition or tensor submission.
    pub fn validate(&self) -> Result<(), QsaError> {
        let required = self.required_workspace()?;
        if required > self.workspace_bytes {
            return Err(QsaError::Workspace {
                required,
                limit: self.workspace_bytes,
            });
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy)]
struct Candidate {
    score: f32,
    block: usize,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.block == other.block
    }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap's maximum is the worst retained score; equal scores prefer
        // the earlier block deterministically. Reference top-k ties are unspecified.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.block.cmp(&other.block))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Scores complete summaries in bounded tiles, retaining only top-k block IDs.
/// Positions are read only for selected blocks and returned in original causal
/// order, matching the reference's scatter-to-mask attention order. The incomplete
/// visible tail is included even after the complete-block budget is exhausted.
/// Streams belong to one batch lane and contain no padding records.
pub fn select_positions<T: Tensor, K: AppendOnlyStream<T>, P: AppendOnlyStream<T>>(
    spec: QsaSelectionSpec,
    query: &T,
    keys: &mut K,
    positions: &mut P,
    tail: &[i32],
    context: &T::Context,
) -> Result<Vec<i32>, QsaError> {
    spec.validate()?;
    if query.shape() != [spec.heads, spec.dimensions]
        || keys.specification().width != spec.dimensions
        || positions.specification().width != spec.ratio
        || positions.specification().element != TensorElementType::I32
        || keys.len() != positions.len()
        || tail.len() >= spec.ratio as usize
        || tail.iter().any(|p| *p < 0)
        || !tail.windows(2).all(|p| p[0] < p[1])
    {
        return Err(QsaError::Geometry);
    }
    let budget = (spec.token_budget / spec.ratio) as usize;
    let query = query.cast_float(TensorElementType::F32, context)?;
    let mut best = BinaryHeap::<Candidate>::with_capacity(budget.min(keys.len()));
    for start in (0..keys.len()).step_by(spec.tile_blocks as usize) {
        let end = (start + spec.tile_blocks as usize).min(keys.len());
        let tile = keys
            .read(start..end, context)?
            .cast_float(TensorElementType::F32, context)?;
        if tile.shape() != [(end - start) as i32, spec.dimensions] {
            return Err(QsaError::Geometry);
        }
        let scores =
            T::matmul(&query, &tile.transpose(context)?, context)?.maximum_scalar(0., context)?;
        let scores = T::sum_axis(&scores, 0, false, context)?
            .multiply_scalar(1. / (spec.dimensions as f32).sqrt(), context)?
            .to_f32_vec(context)?;
        if scores.len() != end - start {
            return Err(QsaError::Geometry);
        }
        for (offset, score) in scores.into_iter().enumerate() {
            if !score.is_finite() {
                return Err(QsaError::Score);
            }
            let candidate = Candidate {
                score,
                block: start + offset,
            };
            if best.len() < budget {
                best.push(candidate)
            } else if candidate < *best.peek().expect("positive budget") {
                best.pop();
                best.push(candidate);
            }
        }
    }
    let mut selected = Vec::with_capacity((spec.token_budget + spec.ratio - 1) as usize);
    for block in best {
        let values = positions.read(block.block..block.block + 1, context)?;
        if values.shape() != [1, spec.ratio]
            || values.element_type() != Some(TensorElementType::I32)
        {
            return Err(QsaError::Geometry);
        }
        let values = values.to_i32_vec(context)?;
        if values.len() != spec.ratio as usize
            || values.iter().any(|p| *p < 0)
            || !values.windows(2).all(|p| p[0] < p[1])
            || tail
                .first()
                .is_some_and(|first| values.last().is_some_and(|last| last >= first))
        {
            return Err(QsaError::Geometry);
        }
        selected.extend(values);
    }
    selected.extend_from_slice(tail);
    selected.sort_unstable();
    if selected.windows(2).any(|p| p[0] == p[1]) {
        return Err(QsaError::Geometry);
    }
    selected.resize((spec.token_budget + spec.ratio - 1) as usize, -1);
    Ok(selected)
}

/// Bounded tensor selection with optional native ranking and deferred validation.
/// Returns `None` when row-selection mechanisms are unavailable or the history
/// exceeds one admitted tile.
/// Family scoring, tiling, causal positions and padding remain architecture-owned.
pub fn select_positions_tensor<T: Tensor, K: AppendOnlyStream<T>, P: AppendOnlyStream<T>>(
    spec: QsaSelectionSpec,
    query: &T,
    keys: &mut K,
    positions: &mut P,
    tail: &[i32],
    upper: i32,
    context: &T::Context,
) -> Result<Option<T>, QsaError> {
    spec.validate()?;
    if upper < 0
        || query.shape() != [spec.heads, spec.dimensions]
        || keys.specification().width != spec.dimensions
        || positions.specification().width != spec.ratio
        || positions.specification().element != TensorElementType::I32
        || keys.len() != positions.len()
        || tail.len() >= spec.ratio as usize
        || tail.iter().any(|p| *p < 0 || *p >= upper)
        || !tail.windows(2).all(|p| p[0] < p[1])
    {
        return Err(QsaError::Geometry);
    }
    // A lazy chain of tiles would retain unbounded temporary graphs. Keep the
    // native path within one admitted tile; longer histories use the bounded
    // portable selector, which completes each tile before advancing.
    if !T::supports_row_selection(context) || keys.len() > spec.tile_blocks as usize {
        return Ok(None);
    }
    let selected = if keys.len() > 0 {
        let query = query.cast_float(TensorElementType::F32, context)?;
        let tile = keys
            .read(0..keys.len(), context)?
            .cast_float(TensorElementType::F32, context)?;
        let rows = positions.read(0..positions.len(), context)?;
        if tile.shape() != [keys.len() as i32, spec.dimensions]
            || rows.shape() != [keys.len() as i32, spec.ratio]
        {
            return Err(QsaError::Geometry);
        }
        let scores =
            T::matmul(&query, &tile.transpose(context)?, context)?.maximum_scalar(0., context)?;
        let scores = T::sum_axis(&scores, 0, false, context)?
            .multiply_scalar(1. / (spec.dimensions as f32).sqrt(), context)?;
        let Some((_, rows)) = scores.topk_rows(&rows, spec.token_budget / spec.ratio, context)?
        else {
            return Ok(None);
        };
        let Some(selected) =
            rows.sorted_unique_indices(tail.first().copied().unwrap_or(upper), context)?
        else {
            return Ok(None);
        };
        selected
    } else {
        T::from_i32_slice(&[], &[0], context)?
    };
    let used = selected.dim(0) + tail.len() as i32;
    let slots = spec.token_budget + spec.ratio - 1;
    let tail = T::from_i32_slice(tail, &[tail.len() as i32], context)?;
    let padding = T::from_i32_slice(&vec![-1; (slots - used) as usize], &[slots - used], context)?;
    Ok(Some(T::concatenate(
        &[selected, tail, padding],
        0,
        context,
    )?))
}

/// Projection, normalization and rotary declarations for one QSA indexer.
#[derive(Debug, Clone)]
pub struct QsaIndexerSpec {
    /// Complete-block selection geometry and workspace.
    pub selection: QsaSelectionSpec,
    /// Joint query/key projection in component-major order.
    pub query_key: LinearSpec,
    /// Query normalization over one index head.
    pub query_norm: NormalizationConstructionSpec,
    /// Pooled-key normalization over one index head.
    pub key_norm: NormalizationConstructionSpec,
    /// Exact shared rotary dimensions, base and product-rounding policy.
    pub rotary: RotarySpec,
}
impl QsaIndexerSpec {
    /// Context-free geometry and format validation before backend allocation.
    pub fn validate(&self) -> Result<(), QsaError> {
        self.selection.validate()?;
        let width = self
            .selection
            .heads
            .checked_add(1)
            .and_then(|h| h.checked_mul(self.selection.dimensions))
            .ok_or(QsaError::Geometry)?;
        if self.query_key.input <= 0
            || self.query_key.output != width
            || self.query_key.bias.is_some()
            || self.rotary.dimensions <= 0
            || self.rotary.dimensions % 2 != 0
            || self.rotary.dimensions > self.selection.dimensions
            || !self.rotary.base.is_finite()
            || self.rotary.base <= 0.
        {
            return Err(QsaError::Geometry);
        }
        self.rotary.algorithm.validate()?;
        for norm in [&self.query_norm, &self.key_norm] {
            norm.validate()?;
            if norm.dimensions != self.selection.dimensions || norm.groups.is_some() {
                return Err(QsaError::Geometry);
            }
        }
        self.query_key
            .format
            .validate_for_weight(&self.query_key.weight)?;
        Ok(())
    }
}
/// Family-owned indexer equations composed from generic neural primitives.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct QsaIndexer<B: NeuralBackend> {
    #[parameter(skip)]
    selection: QsaSelectionSpec,
    query_key: B::Linear,
    query_norm: B::Normalization,
    key_norm: B::Normalization,
    rotary: B::Rotary,
}
impl<B: NeuralBackend> QsaIndexer<B> {
    /// Validates construction before allocating unloaded parameters.
    pub fn new(
        spec: QsaIndexerSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, QsaError> {
        spec.validate()?;
        B::require_operator_capabilities(
            "qwen4_exp QSA",
            C::CAST_FLOAT.union(C::TO_F32_VEC).union(C::TO_I32_VEC),
        )?;
        Ok(Self {
            selection: spec.selection,
            query_key: B::linear(spec.query_key, context)?,
            query_norm: B::normalization(spec.query_norm, context)?,
            key_norm: B::normalization(spec.key_norm, context)?,
            rotary: B::rotary(spec.rotary, context)?,
        })
    }
    /// Produces normalized/rotated queries `[batch,heads,tokens,index_dim]`
    /// and unnormalized raw token keys `[batch,tokens,index_dim]`.
    pub fn project(
        &mut self,
        input: &B::Tensor,
        position: RotaryPosition<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), QsaError> {
        if input.shape().len() != 3 || input.dim(0) <= 0 || input.dim(1) <= 0 {
            return Err(QsaError::Geometry);
        }
        let projected = self.query_key.forward(input, context)?;
        let qw = self.selection.heads * self.selection.dimensions;
        let query = projected
            .index(&[Index::Full, Index::Full, Index::Range(0, qw)], context)?
            .reshape(
                &[
                    input.dim(0),
                    input.dim(1),
                    self.selection.heads,
                    self.selection.dimensions,
                ],
                context,
            )?;
        let query = self
            .query_norm
            .forward(&query, context)?
            .swap_axes(1, 2, context)?;
        let query = self.rotary.forward(&query, position, context)?;
        let keys = projected.index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(qw, qw + self.selection.dimensions),
            ],
            context,
        )?;
        Ok((query, keys))
    }
    /// Caches each complete visible micro-block once. Mean reduction is FP32,
    /// cast back before RMS normalization and rotary at its first original position.
    pub fn summarize(
        &mut self,
        raw_keys: &B::Tensor,
        first_position: RotaryPosition<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, QsaError> {
        if raw_keys.shape() != [self.selection.ratio, self.selection.dimensions] {
            return Err(QsaError::Geometry);
        }
        let dtype = raw_keys.element_type().ok_or(QsaError::Geometry)?;
        let pooled = B::Tensor::mean_axis(
            &raw_keys.cast_float(TensorElementType::F32, context)?,
            0,
            true,
            context,
        )?
        .cast_float(dtype, context)?;
        let key = self
            .key_norm
            .forward(&pooled, context)?
            .reshape(&[1, 1, 1, self.selection.dimensions], context)?;
        Ok(self
            .rotary
            .forward(&key, first_position, context)?
            .reshape(&[1, self.selection.dimensions], context)?)
    }
}

/// First-token rotary values retained while a visible micro-block is incomplete.
#[derive(Debug, Clone)]
pub enum QsaBlockPosition<T> {
    /// Ordinary text positions.
    Offset(i32),
    /// Shared media-position embeddings copied only for the block's first token.
    Embeddings {
        /// Cosine tensor for one token.
        cosine: T,
        /// Sine tensor for one token.
        sine: T,
    },
}
impl<T> QsaBlockPosition<T> {
    /// Borrows the ordinary reusable rotary input contract.
    pub fn as_position(&self) -> RotaryPosition<'_, T> {
        match self {
            Self::Offset(offset) => RotaryPosition::Offset(*offset),
            Self::Embeddings { cosine, sine } => RotaryPosition::Embeddings { cosine, sine },
        }
    }
}
/// Completed block proposed for summary/position stream append.
#[derive(Debug, Clone)]
pub struct QsaCompleteBlock<T> {
    /// Exactly `ratio` raw index keys, in visible-token order.
    pub raw_keys: T,
    /// Exact original K/V positions (padding gaps are preserved).
    pub positions: Vec<i32>,
    /// Rotary source for the first visible token.
    pub first_position: QsaBlockPosition<T>,
}
/// Bounded working representation of one lane's incomplete block.
/// Session ownership uses `QsaPartialStateSpec` to publish every field into the
/// ordinary declared state; this host view is reconstructed at invocation boundaries.
#[derive(Debug, Clone)]
pub struct QsaPartialBlock<T: Tensor> {
    ratio: i32,
    dimensions: i32,
    raw: Option<T>,
    positions: Vec<i32>,
    first_position: Option<QsaBlockPosition<T>>,
    last_seen: Option<i32>,
}
impl<T: Tensor> QsaPartialBlock<T> {
    /// Creates empty logical state. At most `ratio - 1` raw keys are retained.
    pub fn new(ratio: i32, dimensions: i32) -> Result<Self, QsaError> {
        if ratio <= 0 || dimensions <= 0 {
            return Err(QsaError::Geometry);
        }
        Ok(Self {
            ratio,
            dimensions,
            raw: None,
            positions: Vec::new(),
            first_position: None,
            last_seen: None,
        })
    }
    /// Original incomplete-tail K/V positions supplied to every query's selection.
    pub fn tail(&self) -> &[i32] {
        &self.positions
    }
    /// Last consumed original token position, including padding.
    pub fn last_seen(&self) -> Option<i32> {
        self.last_seen
    }
    /// Exact tensor roots retained by completion/snapshot owners.
    pub fn retained_values(&self) -> Vec<&T> {
        let mut values: Vec<&T> = self.raw.iter().collect();
        if let Some(QsaBlockPosition::Embeddings { cosine, sine }) = &self.first_position {
            values.extend([cosine, sine]);
        }
        values
    }
    /// Accumulates one visible key without grouping padding. A complete block is
    /// emitted once; the caller appends its normalized summary and exact positions
    /// under the same surrounding state transaction. Failure leaves this partial
    /// state untouched. No history-wide tensor or causal mask is constructed.
    pub fn push(
        &mut self,
        key: &T,
        position: i32,
        visible: bool,
        rotary: RotaryPosition<'_, T>,
        context: &T::Context,
    ) -> Result<Option<QsaCompleteBlock<T>>, QsaError> {
        if key.shape() != [1, self.dimensions]
            || position < 0
            || self.last_seen.is_some_and(|last| position <= last)
            || !matches!(
                key.element_type(),
                Some(
                    TensorElementType::F16
                        | TensorElementType::Bf16
                        | TensorElementType::F32
                        | TensorElementType::F64
                )
            )
        {
            return Err(QsaError::Geometry);
        }
        if !visible {
            self.last_seen = Some(position);
            return Ok(None);
        }
        let raw = match &self.raw {
            None => key.compact(context)?,
            Some(raw) => {
                if raw.element_type() != key.element_type() {
                    return Err(QsaError::Geometry);
                }
                T::concatenate(&[raw.clone(), key.clone()], 0, context)?.compact(context)?
            }
        };
        let first = if let Some(first) = &self.first_position {
            first.clone()
        } else {
            match rotary {
                RotaryPosition::Offset(offset) => {
                    if offset < 0 {
                        return Err(QsaError::Geometry);
                    }
                    QsaBlockPosition::Offset(offset)
                }
                RotaryPosition::Embeddings { cosine, sine } => {
                    if cosine.shape() != sine.shape()
                        || cosine.shape().iter().rev().skip(1).any(|size| *size != 1)
                        || cosine
                            .shape()
                            .last()
                            .is_none_or(|size| *size <= 0 || *size > self.dimensions)
                    {
                        return Err(QsaError::Geometry);
                    }
                    QsaBlockPosition::Embeddings {
                        cosine: cosine.compact(context)?,
                        sine: sine.compact(context)?,
                    }
                }
            }
        };
        let mut positions = self.positions.clone();
        positions.push(position);
        if positions.len() == self.ratio as usize {
            self.raw = None;
            self.positions.clear();
            self.first_position = None;
            self.last_seen = Some(position);
            Ok(Some(QsaCompleteBlock {
                raw_keys: raw,
                positions,
                first_position: first,
            }))
        } else {
            self.raw = Some(raw);
            self.positions = positions;
            self.first_position = Some(first);
            self.last_seen = Some(position);
            Ok(None)
        }
    }
}
