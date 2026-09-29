//! Exact host-integer n-gram selection from retained original token IDs.
//! Equations follow Transformers 27166ea03f12c940f23176a904ab1d2ff1a3dcbb.

/// Exact checkpoint constants for every order and hash head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGramHashSpec {
    vocabulary: u64,
    eos: u64,
    order: usize,
    heads: usize,
    multipliers: Vec<i64>,
    moduli: Vec<i64>,
    offsets: Vec<i64>,
    table_rows: u64,
}

/// Malformed checkpoint buffers, input geometry or missing original IDs.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NGramError {
    /// Constants do not describe the declared table and hash geometry.
    #[error("invalid n-gram hash geometry or exact integer companions")]
    Constants,
    /// Embedding-only requests must carry original token IDs explicitly.
    #[error("n-gram injection requires original token IDs alongside embeddings")]
    MissingTokenIds,
    /// Batch, sequence and retained history dimensions differ.
    #[error("n-gram input or retained token history shape differs from the declared geometry")]
    Shape,
    /// A token is outside the declared ordinary vocabulary.
    #[error("original token {0} is outside the declared vocabulary")]
    Token(u64),
    /// Hash outputs exceed the admitted lookup request count.
    #[error("n-gram lookup requires {required} rows but admits {limit}")]
    Budget {
        /// Exact output row count.
        required: usize,
        /// Admitted maximum count.
        limit: usize,
    },
}

/// Pure proposal: the execution owner commits history only after its unit succeeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGramRows {
    /// Table rows in `[batch, tokens, (order - 1) * heads]` order.
    pub rows: Vec<u64>,
    /// Original unmodified tokens in `[batch, order - 1]` order.
    pub next_history: Vec<u64>,
}
impl NGramHashSpec {
    pub(super) fn matches_geometry(
        &self,
        vocabulary: u64,
        eos: u64,
        order: usize,
        heads: usize,
    ) -> bool {
        self.vocabulary == vocabulary
            && self.eos == eos
            && self.order == order
            && self.heads == heads
    }
    /// Validates exact signed int64 checkpoint buffers without floating conversion.
    /// Vocabulary primes and multipliers are consumed literally; no reconstruction
    /// from checkpoint names or rounded tensor values is permitted.
    pub fn new(
        vocabulary: u64,
        eos: u64,
        order: usize,
        heads: usize,
        multipliers: Vec<i64>,
        moduli: Vec<i64>,
        offsets: Vec<i64>,
        table_rows: u64,
    ) -> Result<Self, NGramError> {
        let count = order
            .checked_sub(1)
            .and_then(|n| n.checked_mul(heads))
            .filter(|n| *n > 0)
            .ok_or(NGramError::Constants)?;
        if vocabulary == 0
            || vocabulary > i64::MAX as u64
            || eos >= vocabulary
            || order < 2
            || multipliers.len() != order
            || moduli.len() != count
            || offsets.len() != count
            || table_rows == 0
            || table_rows > i64::MAX as u64
        {
            return Err(NGramError::Constants);
        }
        for (&modulus, &offset) in moduli.iter().zip(&offsets) {
            if modulus <= 0
                || offset < 0
                || (offset as u64)
                    .checked_add(modulus as u64)
                    .is_none_or(|end| end > table_rows)
            {
                return Err(NGramError::Constants);
            }
        }
        Ok(Self {
            vocabulary,
            eos,
            order,
            heads,
            multipliers,
            moduli,
            offsets,
            table_rows,
        })
    }
    /// Exact row count of the shared embedding source.
    pub fn table_rows(&self) -> u64 {
        self.table_rows
    }
    /// Fixed retained history width. It is independent of prompt length.
    pub fn history_tokens(&self) -> usize {
        self.order - 1
    }
    /// Number of independently hashed table rows per input token.
    pub fn rows_per_token(&self) -> usize {
        self.moduli.len()
    }
    /// Initial history uses EOS as the reference's segment padding value.
    pub fn initial_history(&self, batch: usize) -> Result<Vec<u64>, NGramError> {
        if batch == 0 {
            return Err(NGramError::Shape);
        }
        let count = batch
            .checked_mul(self.history_tokens())
            .ok_or(NGramError::Shape)?;
        Ok(vec![self.eos; count])
    }
    /// Computes signed wrapping products/XOR and Euclidean remainders exactly.
    /// EOS resets *following* n-grams; it does not terminate generation here.
    pub fn select(
        &self,
        input: Option<&[u64]>,
        batch: usize,
        tokens: usize,
        previous: &[u64],
        max_rows: usize,
    ) -> Result<NGramRows, NGramError> {
        let input = input.ok_or(NGramError::MissingTokenIds)?;
        let count = batch.checked_mul(tokens).ok_or(NGramError::Shape)?;
        let history_count = batch
            .checked_mul(self.history_tokens())
            .ok_or(NGramError::Shape)?;
        if batch == 0 || tokens == 0 || input.len() != count || previous.len() != history_count {
            return Err(NGramError::Shape);
        }
        let required = count
            .checked_mul(self.rows_per_token())
            .ok_or(NGramError::Shape)?;
        if required > max_rows {
            return Err(NGramError::Budget {
                required,
                limit: max_rows,
            });
        }
        for &token in input.iter().chain(previous) {
            if token >= self.vocabulary {
                return Err(NGramError::Token(token));
            }
        }
        let mut rows = Vec::with_capacity(required);
        let mut next_history = Vec::with_capacity(history_count);
        let width = self.history_tokens();
        for b in 0..batch {
            let mut history = previous[b * width..(b + 1) * width].to_vec();
            for &token in &input[b * tokens..(b + 1) * tokens] {
                let mut mixed = (token as i64).wrapping_mul(self.multipliers[0]);
                let mut past_eos = false;
                for lag in 1..self.order {
                    let original = history[width - lag];
                    past_eos |= original == self.eos;
                    let shifted = if past_eos { self.eos } else { original };
                    mixed ^= (shifted as i64).wrapping_mul(self.multipliers[lag]);
                    let head_start = (lag - 1) * self.heads;
                    for head in head_start..head_start + self.heads {
                        rows.push(
                            (mixed.rem_euclid(self.moduli[head]) + self.offsets[head]) as u64,
                        );
                    }
                }
                history.rotate_left(1);
                history[width - 1] = token;
            }
            next_history.extend(history);
        }
        Ok(NGramRows { rows, next_history })
    }
}

/// Reference initialization used only when architecture construction explicitly
/// declares generated multipliers. Stored integer companions take precedence.
pub fn layer_multipliers(
    vocabulary: u64,
    order: usize,
    layer_index: u64,
    seed: u64,
) -> Result<Vec<i64>, NGramError> {
    if vocabulary == 0 || vocabulary > i64::MAX as u64 || order < 2 {
        return Err(NGramError::Constants);
    }
    const GAMMA: u64 = 0x9E3779B97F4A7C15;
    fn splitmix(value: u64) -> u64 {
        let mut value = value.wrapping_add(GAMMA);
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }
    let bound = ((i64::MAX as u64 / vocabulary) / 2).max(1);
    let base = seed.wrapping_add(10007u64.wrapping_mul(layer_index));
    Ok((0..order)
        .map(|i| {
            (2 * (splitmix(base.wrapping_add(GAMMA.wrapping_mul(i as u64 + 1))) % bound) + 1) as i64
        })
        .collect())
}

#[cfg(test)]
mod tests;

/// Header-declared lookup and integer-history geometry, independent of hash literals.
#[derive(Debug, Clone)]
pub struct NGramEmbeddingSpec {
    vocabulary: u64,
    eos: u64,
    order: usize,
    heads: usize,
    table: eredu_runtime::RowLookupSpec,
    history: eredu_runtime::IntegerHistorySpec,
    max_rows: usize,
}
/// Executable n-gram embedding with exact hash literals bound to admitted geometry.
/// This module owns no checkpoint payload or mutable history.
#[derive(Debug, Clone)]
pub struct NGramEmbedding {
    spec: NGramEmbeddingSpec,
    hash: NGramHashSpec,
}
/// Exact input, state or table-provider failure while preparing n-gram embeddings.
#[derive(Debug, thiserror::Error)]
pub enum NGramEmbeddingError {
    /// Explicit injection-unit position or fixed-state publication failure.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),
    /// Integer equation or geometry failure.
    #[error(transparent)]
    Hash(#[from] NGramError),
    /// Bounded state failure.
    #[error(transparent)]
    History(#[from] eredu_runtime::IntegerHistoryError),
    /// Selected source, residency, decode or completion failure.
    #[error(transparent)]
    Lookup(#[from] eredu_runtime::RowLookupError),
    /// Tensor reshape failure.
    #[error(transparent)]
    Tensor(#[from] eredu_nn::Error),
}
impl NGramEmbeddingSpec {
    /// Declares table geometry and resumable integer-history identity using only headers.
    pub fn new(
        vocabulary: u64,
        eos: u64,
        order: usize,
        heads: usize,
        table: eredu_runtime::RowLookupSpec,
        history_slot: u32,
        max_rows: usize,
    ) -> Result<Self, NGramEmbeddingError> {
        table.validate()?;
        let rows_per_token = order
            .checked_sub(1)
            .and_then(|n| n.checked_mul(heads))
            .filter(|n| *n > 0)
            .ok_or(NGramError::Constants)?;
        if vocabulary == 0
            || vocabulary > i32::MAX as u64
            || eos >= vocabulary
            || order < 2
            || order > i32::MAX as usize
            || table.rows > i64::MAX as u64
            || max_rows < rows_per_token
            || max_rows > i32::MAX as usize
            || rows_per_token
                .checked_mul(table.dimensions as usize)
                .is_none_or(|width| width > i32::MAX as usize)
        {
            return Err(NGramError::Constants.into());
        }
        let capacity = i32::try_from(order - 1).map_err(|_| NGramError::Constants)?;
        let history = eredu_runtime::IntegerHistorySpec::new(history_slot, capacity, eos as i32)?;
        Ok(Self {
            vocabulary,
            eos,
            order,
            heads,
            table,
            history,
            max_rows,
        })
    }
    /// Logical row lookup retained by the declaration.
    pub fn lookup_spec(&self) -> &eredu_runtime::RowLookupSpec {
        &self.table
    }
    /// Concatenated feature width produced by lookup.
    pub fn output_width(&self) -> i32 {
        ((self.order - 1) * self.heads) as i32 * self.table.dimensions
    }
    /// Exact reset token used to pad masked lexical input positions.
    pub fn padding_token(&self) -> u64 {
        self.eos
    }
    /// Admitted maximum original token count in a lookup request.
    pub fn max_tokens(&self) -> usize {
        self.max_rows / ((self.order - 1) * self.heads)
    }
    /// Maximum n-gram order, also the lexical convolution dilation.
    pub fn order(&self) -> i32 {
        self.order as i32
    }
    /// State policy attached to the owning injection execution unit.
    pub fn history_policy(&self) -> eredu_core::cache::StateTensorPolicy {
        self.history.policy()
    }
    pub(super) fn validate_hash(&self, hash: &NGramHashSpec) -> Result<(), NGramError> {
        if !hash.matches_geometry(self.vocabulary, self.eos, self.order, self.heads)
            || hash.table_rows != self.table.rows
        {
            return Err(NGramError::Constants);
        }
        Ok(())
    }
}
impl NGramEmbedding {
    /// Binds exact checkpoint literals without changing declared table or state geometry.
    pub fn new(spec: NGramEmbeddingSpec, hash: NGramHashSpec) -> Result<Self, NGramEmbeddingError> {
        spec.validate_hash(&hash)?;
        Ok(Self { spec, hash })
    }
    /// Admitted geometry retained alongside executable hash literals.
    pub fn specification(&self) -> &NGramEmbeddingSpec {
        &self.spec
    }
    /// Produces `[batch, tokens, embedding width]` using exact original IDs.
    /// Provider errors leave history untouched; successful history updates belong
    /// to the surrounding layer/session transaction and its ordinary rollback.
    pub fn forward<B, P, S>(
        &self,
        input: Option<&[u64]>,
        batch: usize,
        tokens: usize,
        state: &mut S,
        provider: &mut P,
        access: eredu_runtime::ParameterBankAccess,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, NGramEmbeddingError>
    where
        B: eredu_nn::GroupedNeuralBackend,
        P: eredu_runtime::ParameterProvider<B>,
        S: eredu_runtime::RuntimeStateComponents<B>,
    {
        use eredu_nn::Tensor;
        let input = input.ok_or(NGramError::MissingTokenIds)?;
        let previous = self.spec.history.read::<B, _>(state, batch, context)?;
        let previous = previous
            .into_iter()
            .map(|value| u64::try_from(value).map_err(|_| NGramError::Shape))
            .collect::<Result<Vec<_>, _>>()?;
        let selected =
            self.hash
                .select(Some(input), batch, tokens, &previous, self.spec.max_rows)?;
        let output = provider.lookup_rows(&self.spec.table, &selected.rows, access, context)?;
        let shape = [
            i32::try_from(batch).map_err(|_| NGramError::Shape)?,
            i32::try_from(tokens).map_err(|_| NGramError::Shape)?,
            self.spec.output_width(),
        ];
        let output = output.reshape(&shape, context)?;
        let next = selected
            .next_history
            .into_iter()
            .map(|token| token as i32)
            .collect::<Vec<_>>();
        self.spec
            .history
            .write::<B, _>(state, &next, batch, context)?;
        Ok(output)
    }
}
