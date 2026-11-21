//! Portable resident key/value cache ownership and append policy.

use eredu_core::{
    cache::mechanisms::{ResidentAppendBacking, ResidentAppendStorage, ResidentKeyValueStorage},
    BackendFailure,
};
use eredu_nn::NeuralBackend;

type Storage<M> = <M as eredu_core::cache::mechanisms::LiveCacheBackend>::Tensor;
type Queue<M> = <M as eredu_core::cache::mechanisms::LiveCacheBackend>::Queue;

fn invalid(detail: impl Into<String>) -> BackendFailure {
    BackendFailure::from_error(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        detail.into(),
    ))
}

/// A cache that appends all key/value states along the sequence axis.
#[derive(Debug, Clone)]
pub struct ResidentKeyValueCache<M: ResidentKeyValueStorage> {
    keys: Option<Storage<M>>,
    values: Option<Storage<M>>,
    key_only: bool,
    append_only: bool,
    key_buffer: Option<M::Prefix>,
    value_buffer: Option<M::Prefix>,
    offset: i32,
    length: i32,
    capacity: i32,
    step: i32,
    max_size: Option<i32>,
    /// Attention window whose retained past is one token shorter because the
    /// current token is included in the window. Updates return the full
    /// submitted span before applying this retention bound.
    attention_window: Option<i32>,
}

impl<M: ResidentKeyValueStorage> Default for ResidentKeyValueCache<M> {
    fn default() -> Self {
        Self {
            keys: None,
            values: None,
            key_only: false,
            append_only: false,
            key_buffer: None,
            value_buffer: None,
            offset: 0,
            length: 0,
            capacity: 0,
            step: 0,
            max_size: None,
            attention_window: None,
        }
    }
}

impl<M: ResidentKeyValueStorage> ResidentKeyValueCache<M> {
    fn append_eligible(&self, additional: u64) -> bool {
        self.append_only
            && self.attention_window.is_none_or(|window| {
                self.offset == self.length
                    && u64::try_from(self.length)
                        .ok()
                        .and_then(|length| length.checked_add(additional))
                        .is_some_and(|required| required < window as u64)
            })
    }

    /// Maximum logical sequence dimension retained after the admitted input
    /// span. Includes chunk padding and existing capacity; sliding views are
    /// priced by their logical extent, not the allocator's underlying buffer.
    pub fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        let required = u64::try_from(self.length).ok()?.checked_add(additional)?;
        let bound = if let Some(window) = self
            .attention_window
            .filter(|_| !self.append_eligible(additional))
        {
            required.min(u64::try_from(window.checked_sub(1)?).ok()?)
        } else {
            let step = u64::try_from(self.step.max(1)).ok()?;
            let padded = required
                .checked_add(step - 1)?
                .checked_div(step)?
                .checked_mul(step)?;
            match (self.attention_window, self.max_size) {
                (Some(window), _) => padded.min(u64::try_from(window.checked_sub(1)?).ok()?),
                (_, Some(maximum)) => padded.min(u64::try_from(maximum).ok()?),
                _ => padded,
            }
        };
        Some(bound.max(u64::try_from(self.capacity).ok()?))
    }

    /// Independent snapshot of retained elements, including strided/windowed views.
    /// Private append capacity is discarded; offsets and window metadata persist.
    pub fn isolated_snapshot(&self, stream: &Queue<M>) -> Result<Self, BackendFailure> {
        let mut cache = self.clone();
        cache.keys = self
            .keys
            .as_ref()
            .map(|array| M::copy_tensor(array, stream))
            .transpose()?;
        cache.values = self
            .values
            .as_ref()
            .map(|array| M::copy_tensor(array, stream))
            .transpose()?;
        if cache.append_only {
            cache.key_buffer = None;
            cache.value_buffer = None;
            cache.capacity = cache.length;
        }
        Ok(cache)
    }

    /// Creates a cache with private chunked capacity and immutable visible prefixes.
    /// Divergent descendants allocate independently; committed appends reuse only
    /// unused suffix slots. Capacity grows in token chunks.
    pub fn new_append_only() -> Self {
        Self {
            append_only: true,
            step: 128,
            ..Self::default()
        }
    }

    /// Append-only capacity cache with shared key/value data.
    pub fn new_key_only_append_only() -> Self {
        Self {
            key_only: true,
            ..Self::new_append_only()
        }
    }

    /// Sliding cache with private suffix reuse before the first retention crop.
    /// Once history is discarded, ordinary immutable concatenation preserves
    /// absolute positions and returns the complete newly submitted span.
    pub fn new_append_only_for_sliding_attention(window: i32) -> Self {
        assert!(window > 0, "sliding attention window must be positive");
        Self {
            attention_window: Some(window),
            ..Self::new_append_only()
        }
    }

    /// Key-only sliding cache with private suffix reuse before retention begins.
    pub fn new_key_only_append_only_for_sliding_attention(window: i32) -> Self {
        Self {
            key_only: true,
            ..Self::new_append_only_for_sliding_attention(window)
        }
    }

    /// Private capacity not represented by visible retained tensor extents.
    pub fn spare_capacity_bytes(&self) -> Option<u64> {
        [self.key_buffer.as_ref(), self.value_buffer.as_ref()]
            .into_iter()
            .flatten()
            .try_fold(0u64, |sum, buffer| {
                sum.checked_add(M::prefix_spare_bytes(buffer))
            })
    }

    /// Creates an empty concatenating key/value cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Copies the committed arrays into independent native storage.
    pub fn deep_clone_state(&self) -> Result<Self, BackendFailure> {
        let mut cache = self.clone();
        cache.keys = self.keys.as_ref().map(M::fork_tensor).transpose()?;
        cache.values = self.values.as_ref().map(M::fork_tensor).transpose()?;
        if cache.append_only {
            cache.key_buffer = None;
            cache.value_buffer = None;
            cache.capacity = cache.length;
        }
        Ok(cache)
    }

    /// Reports whether published prefix handles can be shared with a checkpoint.
    /// Chunked overwrite backing requires physical isolation.
    pub fn has_immutable_roots(&self) -> bool {
        self.append_only || self.step <= 1
    }

    /// Shares immutable prefixes without changing array layout. Reserved suffixes
    /// force divergent descendants to allocate independently. Overwriting storage
    /// requires a physical copy.
    pub fn checkpoint_clone_state(&self) -> Result<Self, BackendFailure> {
        if self.has_immutable_roots() {
            Ok(self.clone())
        } else {
            self.deep_clone_state()
        }
    }

    /// Creates a cache for exact causal sliding-window attention.
    ///
    /// A window of `N` includes the current token, so the cache retains at most
    /// `N - 1` past states between calls while returning every newly submitted
    /// state for multi-token prefill attention.
    pub fn new_for_sliding_attention(window: i32) -> Self {
        assert!(window > 0, "sliding attention window must be positive");
        Self {
            attention_window: Some(window),
            ..Self::default()
        }
    }

    /// Creates an exact sliding-window cache that persists keys only.
    pub fn new_key_only_for_sliding_attention(window: i32) -> Self {
        assert!(window > 0, "sliding attention window must be positive");
        Self {
            key_only: true,
            attention_window: Some(window),
            ..Self::default()
        }
    }

    /// Creates an unbounded cache whose backing arrays grow in `step`-token chunks.
    pub fn new_with_step(step: i32) -> Self {
        Self {
            step: step.max(1),
            ..Self::default()
        }
    }

    /// Creates a bounded cache whose backing arrays grow in `step`-token chunks.
    pub fn new_with_max_size_and_step(max_size: i32, step: i32) -> Self {
        Self {
            max_size: Some(max_size),
            step: step.max(1),
            ..Self::default()
        }
    }

    /// Truncates the cache to `len` tokens.
    fn truncate_candidate(&mut self, len: i32, stream: &Queue<M>) -> Result<(), BackendFailure> {
        if self.attention_window.is_some() {
            if len < 0 || len > self.offset {
                return Err(invalid(format!(
                    "sliding cache truncate position {len} is outside 0..{}",
                    self.offset
                )));
            }
            if len == 0 {
                self.clear();
                return Ok(());
            }
            let retained_origin = self.offset - self.length;
            if len < retained_origin {
                return Err(invalid(format!(
                    "cannot roll a sliding cache back to absolute position {len}; retained state starts at {retained_origin}"
                )));
            }
            let retained = len - retained_origin;
            if let Some(keys) = self.keys.take() {
                self.keys = Some(M::slice_tokens(&keys, 2, 0..retained, stream)?);
            }
            if let Some(values) = self.values.take() {
                self.values = Some(M::slice_tokens(&values, 2, 0..retained, stream)?);
            }
            self.key_buffer = self
                .key_buffer
                .as_ref()
                .map(|b| M::truncate_prefix(b, retained))
                .transpose()?;
            self.value_buffer = self
                .value_buffer
                .as_ref()
                .map(|b| M::truncate_prefix(b, retained))
                .transpose()?;
            self.offset = len;
            self.length = retained;
            if self.key_buffer.is_none() {
                self.capacity = retained;
            }
            return Ok(());
        }
        if len < 0 || len > self.length {
            return Err(invalid(format!(
                "concatenating cache truncate length {len} is outside 0..{}",
                self.length
            )));
        }
        if self.append_only {
            let key_buffer = self
                .key_buffer
                .as_ref()
                .map(|b| M::truncate_prefix(b, len))
                .transpose()?;
            let value_buffer = self
                .value_buffer
                .as_ref()
                .map(|b| M::truncate_prefix(b, len))
                .transpose()?;
            let keys = self
                .keys
                .as_ref()
                .map(|a| M::slice_tokens(a, 2, 0..len, stream))
                .transpose()?;
            let values = self
                .values
                .as_ref()
                .map(|a| M::slice_tokens(a, 2, 0..len, stream))
                .transpose()?;
            self.key_buffer = key_buffer;
            self.value_buffer = value_buffer;
            self.keys = keys;
            self.values = values;
            self.offset = len;
            self.length = len;
            return Ok(());
        }
        if self.step > 1 {
            // Chunked caches treat the array shape as backing capacity and
            // expose only `length` through logical_arrays. Rolling the logical
            // frontier back is sufficient: the next update overwrites the
            // abandoned speculative range in place without forcing capacity
            // regrowth.
            self.offset = len;
            self.length = len;
            return Ok(());
        }
        if let Some(keys) = self.keys.take() {
            self.keys = Some(M::slice_tokens(&keys, 2, 0..len, stream)?);
        }
        if let Some(values) = self.values.take() {
            self.values = Some(M::slice_tokens(&values, 2, 0..len, stream)?);
        }
        self.offset = len;
        self.length = len;
        self.capacity = len;
        Ok(())
    }

    /// Returns the arrays currently retained by the cache.
    pub fn arrays(&self) -> impl Iterator<Item = &Storage<M>> {
        self.keys.iter().chain(self.values.iter())
    }

    /// Clears cached arrays while preserving cache configuration.
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
        self.key_buffer = None;
        self.value_buffer = None;
        self.offset = 0;
        self.length = 0;
        self.capacity = 0;
    }
}

impl<M: ResidentKeyValueStorage> ResidentKeyValueCache<M> {
    fn grown_capacity(&self, required: i32) -> Result<i32, BackendFailure> {
        let step = self.step.max(1);
        let capacity = required
            .checked_add(step - 1)
            .and_then(|rounded| (rounded / step).checked_mul(step))
            .ok_or_else(|| invalid("cache capacity overflow"))?;
        Ok(self
            .max_size
            .map_or(capacity, |maximum| capacity.min(maximum)))
    }

    fn update_append_only(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        new_tokens: i32,
        stream: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), BackendFailure> {
        let required = self
            .length
            .checked_add(new_tokens)
            .ok_or_else(|| invalid("cache append overflow"))?;
        let capacity = self
            .continuation_capacity_bound(new_tokens as u64)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| invalid("cache capacity overflow"))?;
        let candidate =
            |buffer: &Option<M::Prefix>, previous: &Option<Storage<M>>, update: &Storage<M>| {
                if let Some(buffer) = buffer {
                    M::append_prefix(buffer, update, capacity, stream)
                } else if let Some(previous) = previous {
                    let prefix = M::new_prefix(previous, capacity, stream)?;
                    M::append_prefix(&prefix, update, capacity, stream)
                } else {
                    M::new_prefix(update, capacity, stream)
                }
            };
        // Publish only after both candidate constructions succeed. Reservations
        // survive failure; another descendant then copies its immutable prefix.
        let key_buffer = candidate(&self.key_buffer, &self.keys, &keys)?;
        let value_buffer = if self.key_only {
            None
        } else {
            Some(candidate(&self.value_buffer, &self.values, &values)?)
        };
        let keys = M::prefix_view(&key_buffer, stream)?;
        let values = match &value_buffer {
            Some(buffer) => M::prefix_view(buffer, stream)?,
            None => keys.clone(),
        };
        self.capacity = M::prefix_capacity(&key_buffer);
        self.key_buffer = Some(key_buffer);
        self.value_buffer = value_buffer;
        self.keys = Some(keys.clone());
        self.values = (!self.key_only).then_some(values.clone());
        self.length = required;
        self.offset = required;
        Ok((keys, values))
    }

    fn logical_arrays(
        &self,
        stream: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), BackendFailure> {
        let keys = M::slice_tokens(
            self.keys.as_ref().expect("cache keys initialized"),
            2,
            0..self.length,
            stream,
        )?;
        let values = if self.key_only {
            keys.clone()
        } else {
            M::slice_tokens(
                self.values.as_ref().expect("cache values initialized"),
                2,
                0..self.length,
                stream,
            )?
        };
        Ok((keys, values))
    }

    /// Borrows logical key and value views for the committed extent.
    pub fn snapshot_arrays(
        &self,
        stream: &Queue<M>,
    ) -> Result<Option<(Storage<M>, Storage<M>)>, BackendFailure> {
        if self.keys.is_none() {
            return Ok(None);
        }
        self.logical_arrays(stream).map(Some)
    }

    /// Installs a restored logical pair at the supplied absolute frontier.
    pub fn restore_resident(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        end: i32,
    ) -> Result<(), BackendFailure> {
        let key_shape = M::shape(&keys);
        let value_shape = M::shape(&values);
        let compatible =
            key_shape.len() == 4 && value_shape.len() == 4 && key_shape[..3] == value_shape[..3];
        let length = key_shape.get(2).copied().unwrap_or(-1);
        if !compatible || end < length {
            return Err(invalid(
                "restored key/value cache arrays do not match their retained range",
            ));
        }
        if self.attention_window.is_none() && end != length {
            return Err(invalid(
                "restored full-attention cache must begin at token zero",
            ));
        }
        if let Some(window) = self.attention_window {
            if length > window.saturating_sub(1) {
                return Err(invalid(
                    "restored sliding cache exceeds its retained history",
                ));
            }
        }
        self.keys = Some(keys);
        self.values = (!self.key_only).then_some(values);
        self.key_buffer = None;
        self.value_buffer = None;
        self.offset = end;
        self.length = length;
        self.capacity = length;
        Ok(())
    }
}

impl<M: ResidentKeyValueStorage> ResidentKeyValueCache<M> {
    fn append_candidate(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), BackendFailure> {
        let key_shape = M::shape(&keys);
        let value_shape = M::shape(&values);
        if key_shape.len() != 4 || value_shape.len() != 4 || key_shape[..3] != value_shape[..3] {
            return Err(invalid(
                "cache append requires matching batch/head/token geometry",
            ));
        }
        let new_tokens = key_shape[2];
        if self.append_eligible(new_tokens as u64) {
            return self.update_append_only(keys, values, new_tokens, stream);
        }
        self.offset = self
            .offset
            .checked_add(new_tokens)
            .ok_or_else(|| invalid("cache frontier overflow"))?;

        if let Some(window) = self.attention_window {
            // Cropping changes the prefix origin. No capacity owner may reserve
            // suffixes relative to the old origin after retention begins.
            self.key_buffer = None;
            self.value_buffer = None;
            let combined_keys = match self.keys.take() {
                Some(previous) => M::concatenate_tokens(&[&previous, &keys], 2, stream)?,
                None => keys,
            };
            let combined_values = if self.key_only {
                combined_keys.clone()
            } else {
                match self.values.take() {
                    Some(previous) => M::concatenate_tokens(&[&previous, &values], 2, stream)?,
                    None => values,
                }
            };
            let retained = (window - 1).min(M::shape(&combined_keys)[2]);
            if retained == 0 {
                self.keys = None;
                self.values = None;
            } else {
                let start = M::shape(&combined_keys)[2] - retained;
                self.keys = Some(M::slice_tokens(
                    &combined_keys,
                    2,
                    start..M::shape(&combined_keys)[2],
                    stream,
                )?);
                self.values = if self.key_only {
                    None
                } else {
                    Some(M::slice_tokens(
                        &combined_values,
                        2,
                        start..M::shape(&combined_values)[2],
                        stream,
                    )?)
                };
            }
            self.length = retained;
            self.capacity = retained;
            return Ok((combined_keys, combined_values));
        }

        if self.step <= 1 {
            if self.key_only {
                self.keys = Some(match self.keys.take() {
                    Some(previous) => M::concatenate_tokens(&[&previous, &keys], 2, stream)?,
                    None => keys,
                });
            } else {
                match (self.keys.take(), self.values.take()) {
                    (Some(k), Some(v)) => {
                        self.keys = Some(M::concatenate_tokens(&[&k, &keys], 2, stream)?);
                        self.values = Some(M::concatenate_tokens(&[&v, &values], 2, stream)?);
                    }
                    _ => {
                        self.keys = Some(keys);
                        self.values = Some(values);
                    }
                }
            }
            if let Some(max_size) = self.max_size {
                let length = M::shape(self.keys.as_ref().expect("cache keys initialized"))[2];
                if length > max_size {
                    let start = length - max_size;
                    let keys = self.keys.as_ref().expect("cache keys initialized");
                    self.keys = Some(M::slice_tokens(keys, 2, start..length, stream)?);
                    if let Some(values) = &self.values {
                        self.values = Some(M::slice_tokens(values, 2, start..length, stream)?);
                    }
                }
            }
            self.length = M::shape(self.keys.as_ref().expect("cache keys initialized"))[2];
            self.capacity = self.length;
            return Ok((
                self.keys.clone().expect("Keys cannot be None"),
                if self.key_only {
                    self.keys.clone().expect("Keys cannot be None")
                } else {
                    self.values.clone().expect("Values cannot be None")
                },
            ));
        }

        let required = self
            .length
            .checked_add(new_tokens)
            .ok_or_else(|| invalid("cache length overflow"))?;

        if let Some(max_size) = self.max_size {
            if required > max_size {
                if self.keys.is_none() {
                    let start = new_tokens - max_size;
                    self.keys = Some(M::slice_tokens(&keys, 2, start..new_tokens, stream)?);
                    if !self.key_only {
                        self.values = Some(M::slice_tokens(&values, 2, start..new_tokens, stream)?);
                    }
                    self.length = max_size;
                    self.capacity = max_size;
                    return self.logical_arrays(stream);
                }
                let (old_keys, old_values) = self.logical_arrays(stream)?;
                let combined_keys = M::concatenate_tokens(&[&old_keys, &keys], 2, stream)?;
                let combined_values = M::concatenate_tokens(&[&old_values, &values], 2, stream)?;
                let start = required - max_size;
                self.keys = Some(M::slice_tokens(
                    &combined_keys,
                    2,
                    start..M::shape(&combined_keys)[2],
                    stream,
                )?);
                if !self.key_only {
                    self.values = Some(M::slice_tokens(
                        &combined_values,
                        2,
                        start..M::shape(&combined_values)[2],
                        stream,
                    )?);
                }
                self.length = max_size;
                self.capacity = max_size;
                return self.logical_arrays(stream);
            }
        }

        if self.keys.is_none() {
            self.capacity = self.grown_capacity(required)?;
            if self.capacity == required {
                self.keys = Some(keys);
                if !self.key_only {
                    self.values = Some(values);
                }
                self.length = required;
                return self.logical_arrays(stream);
            }
            self.keys = Some(M::allocate_tokens(&keys, 2, self.capacity, stream)?);
            if !self.key_only {
                self.values = Some(M::allocate_tokens(&values, 2, self.capacity, stream)?);
            }
        } else if required > self.capacity {
            let new_capacity = self.grown_capacity(required)?;
            let padding = new_capacity - self.capacity;
            let key_padding = M::allocate_tokens(&keys, 2, padding, stream)?;
            let value_padding = (!self.key_only)
                .then(|| M::allocate_tokens(&values, 2, padding, stream))
                .transpose()?;
            self.keys = Some(M::concatenate_tokens(
                &[
                    self.keys.as_ref().expect("cache keys initialized"),
                    &key_padding,
                ],
                2,
                stream,
            )?);
            if let Some(value_padding) = value_padding {
                self.values = Some(M::concatenate_tokens(
                    &[
                        self.values.as_ref().expect("cache values initialized"),
                        &value_padding,
                    ],
                    2,
                    stream,
                )?);
            }
            self.capacity = new_capacity;
        }

        self.keys = Some(M::write_tokens(
            self.keys.take().expect("cache keys initialized"),
            2,
            self.length..required,
            &keys,
            stream,
        )?);
        if !self.key_only {
            self.values = Some(M::write_tokens(
                self.values.take().expect("cache values initialized"),
                2,
                self.length..required,
                &values,
                stream,
            )?);
        }
        self.length = required;
        self.logical_arrays(stream)
    }
}

impl<M: ResidentKeyValueStorage> ResidentKeyValueCache<M> {
    /// Logical number of retained tokens.
    pub fn length(&self) -> i32 {
        self.length
    }
    /// Capacity in tokens of the current resident backing.
    pub fn capacity(&self) -> i32 {
        self.capacity
    }
    /// Absolute token frontier.
    pub fn offset(&self) -> i32 {
        self.offset
    }
    /// Attention window or retained capacity limit, when bounded.
    pub fn max_size(&self) -> Option<i32> {
        self.attention_window.or(self.max_size)
    }
    /// Whether a native prefix reservation still owns suffix capacity.
    pub fn has_prefix_reservation(&self) -> bool {
        self.key_buffer.is_some()
    }

    /// Derives the physical backing selection without submitting work.
    fn append_backing(&self, appended: u64) -> Option<(u64, ResidentAppendBacking)> {
        let visible = u64::try_from(self.length).ok()?.checked_add(appended)?;
        let append_eligible = self.append_eligible(appended);
        let positions = if self.attention_window.is_some() && !append_eligible {
            visible
        } else {
            self.continuation_capacity_bound(appended)?.max(visible)
        };
        let reusable = |prefix: Option<&M::Prefix>| {
            prefix.is_some_and(|prefix| {
                i32::try_from(appended)
                    .is_ok_and(|count| M::can_append_prefix_without_copy(prefix, count))
            })
        };
        let backing = if self.keys.is_none() && positions == appended {
            ResidentAppendBacking::BorrowedInput
        } else if append_eligible
            && reusable(self.key_buffer.as_ref())
            && (self.key_only || reusable(self.value_buffer.as_ref()))
        {
            ResidentAppendBacking::ReservedPrefix
        } else if !self.append_only
            && self.step > 1
            && self.attention_window.is_none()
            && visible <= u64::try_from(self.capacity).ok()?
        {
            ResidentAppendBacking::ExistingCapacity
        } else {
            ResidentAppendBacking::NewAllocation
        };
        Some((positions, backing))
    }

    /// Appends a pair and publishes its frontier only after both values are ready.
    pub fn update_and_fetch(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        queue: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), BackendFailure> {
        let mut candidate = self.clone();
        let output = candidate.append_candidate(keys, values, queue)?;
        *self = candidate;
        Ok(output)
    }

    /// Truncates retained state only after both logical views are prepared.
    pub fn truncate(&mut self, length: i32, queue: &Queue<M>) -> Result<(), BackendFailure> {
        let mut candidate = self.clone();
        candidate.truncate_candidate(length, queue)?;
        *self = candidate;
        Ok(())
    }

    /// Joins a transient pair with retained history without advancing the cache.
    fn visible_with_transient(
        &self,
        keys: &Storage<M>,
        values: &Storage<M>,
        queue: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), BackendFailure> {
        if self.length == 0 {
            return Ok((keys.clone(), values.clone()));
        }
        let (retained_keys, retained_values) = self.logical_arrays(queue)?;
        Ok((
            M::concatenate_tokens(&[&retained_keys, keys], 2, queue)?,
            M::concatenate_tokens(&[&retained_values, values], 2, queue)?,
        ))
    }
}

impl<B, M> crate::RuntimeLayerState<B> for ResidentKeyValueCache<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: ResidentKeyValueStorage,
{
    type RetainedValues<'a> = std::iter::Flatten<std::array::IntoIter<Option<&'a M::Value>, 2>>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        [
            self.keys.as_ref().map(M::borrow_value),
            self.values.as_ref().map(M::borrow_value),
        ]
        .into_iter()
        .flatten()
    }
}

impl<M: ResidentKeyValueStorage> eredu_nn::AttentionCache<M::Value> for ResidentKeyValueCache<M> {
    fn memory_retained_positions(&self) -> Option<u64> {
        u64::try_from(self.length).ok()
    }

    fn update_memory_contract(
        &self,
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_nn::Error> {
        use eredu_nn::{mechanism_memory::MechanismInvocation, Error};
        let MechanismInvocation::CacheUpdate {
            previous,
            appended,
            key_width,
            value_width,
            ..
        } = *invocation
        else {
            return Err(Error::backend(
                "cache update memory requires a cache append invocation",
            ));
        };
        if Some(previous) != self.memory_retained_positions() {
            return Err(Error::backend(
                "cache memory append retained frontier differs from installed state",
            ));
        }
        if self.key_only && key_width != value_width {
            return Err(Error::backend(
                "key-only cache value geometry must match keys",
            ));
        }
        let (positions, backing) = self
            .append_backing(appended)
            .ok_or_else(|| Error::backend("cache append capacity overflowed"))?;
        M::describe_resident_append(
            invocation,
            ResidentAppendStorage {
                positions,
                backing,
                key_only: self.key_only,
                keys: self.keys.as_ref(),
                values: self.values.as_ref(),
            },
        )
    }

    fn offset(&self) -> i32 {
        self.offset()
    }

    fn max_size(&self) -> Option<i32> {
        self.max_size()
    }

    fn update_for_attention(
        &mut self,
        keys: M::Value,
        values: M::Value,
        context: &Queue<M>,
    ) -> Result<(M::Value, M::Value), eredu_nn::Error> {
        let (keys, values) = self
            .update_and_fetch(M::into_storage(keys), M::into_storage(values), context)
            .map_err(eredu_nn::Error::backend_source)?;
        Ok((M::into_value(keys), M::into_value(values)))
    }

    fn indexed_attention<B: NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: eredu_nn::IndexedAttentionInput<'_, M::Value>,
        context: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        request.validate()?;
        B::indexed_attention(request, context)
    }

    fn relative_attention<B: NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: eredu_nn::RelativeAttentionInput<'_, M::Value>,
        context: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        request.validate()?;
        B::relative_attention(request, context)
    }

    fn attention(
        &mut self,
        request: eredu_nn::AttentionRequest<'_, M::Value>,
        context: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        request.validate()?;
        M::resident_attention(request, context)
    }
}

impl<M: ResidentKeyValueStorage> eredu_nn::ReadOnlyAttentionCache<M::Value>
    for ResidentKeyValueCache<M>
{
    fn attention_with_transient(
        &mut self,
        request: eredu_nn::AttentionRequest<'_, M::Value>,
        context: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        use eredu_nn::Error;
        request.validate()?;
        let (keys, values) = self
            .visible_with_transient(
                M::storage_ref(&request.keys),
                M::storage_ref(&request.values),
                context,
            )
            .map_err(Error::backend_source)?;
        M::resident_attention(
            eredu_nn::AttentionRequest {
                keys: M::into_value(keys),
                values: M::into_value(values),
                ..request
            },
            context,
        )
    }
}
