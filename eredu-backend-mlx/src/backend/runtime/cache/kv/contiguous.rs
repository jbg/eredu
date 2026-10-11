//! Contiguous append-only key/value cache storage.

use super::*;
use safemlx::PrefixAppendBuffer;

/// A cache that appends all key/value states along the sequence axis.
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    key_only: bool,
    append_only: bool,
    key_buffer: Option<PrefixAppendBuffer>,
    value_buffer: Option<PrefixAppendBuffer>,
    offset: i32,
    pub(super) length: i32,
    pub(super) capacity: i32,
    step: i32,
    max_size: Option<i32>,
    /// Attention window whose retained past is one token shorter because the
    /// current token is included in the window. Updates return the full
    /// submitted span before applying this retention bound.
    attention_window: Option<i32>,
}

impl ConcatKeyValueCache {
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
    pub(crate) fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
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
    pub(crate) fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        let mut cache = self.clone();
        cache.keys = self
            .keys
            .as_ref()
            .map(|array| array.contiguous(false, stream)?.deep_clone())
            .transpose()?;
        cache.values = self
            .values
            .as_ref()
            .map(|array| array.contiguous(false, stream)?.deep_clone())
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
    /// unused suffix slots. The chunk size is a native allocation policy.
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
    pub(crate) fn spare_capacity_bytes(&self) -> Option<u64> {
        [self.key_buffer.as_ref(), self.value_buffer.as_ref()]
            .into_iter()
            .flatten()
            .try_fold(0u64, |sum, buffer| {
                sum.checked_add(u64::try_from(buffer.spare_capacity_bytes()).ok()?)
            })
    }

    /// Creates an empty concatenating key/value cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty cache that persists keys and reuses them as values.
    pub fn new_key_only() -> Self {
        Self {
            key_only: true,
            ..Self::default()
        }
    }

    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        let mut cache = self.clone();
        cache.keys = self
            .keys
            .as_ref()
            .map(|array| array.clone().deep_clone())
            .transpose()?;
        cache.values = self
            .values
            .as_ref()
            .map(|array| array.clone().deep_clone())
            .transpose()?;
        if cache.append_only {
            cache.key_buffer = None;
            cache.value_buffer = None;
            cache.capacity = cache.length;
        }
        Ok(cache)
    }

    // Published prefixes remain immutable for reserved-suffix and concatenating
    // caches. Overwriting storage needs physical isolation.
    pub(crate) fn has_immutable_roots(&self) -> bool {
        self.append_only || self.step <= 1
    }

    /// Shares immutable prefixes without changing array layout. Reserved suffixes
    /// force divergent descendants to allocate independently. Overwriting storage
    /// requires a physical copy.
    pub fn checkpoint_clone_state(&self) -> Result<Self, Exception> {
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
    #[cfg(test)]
    pub fn new_with_step(step: i32) -> Self {
        Self {
            step: step.max(1),
            ..Self::default()
        }
    }

    /// Creates a bounded cache whose backing arrays grow in `step`-token chunks.
    #[cfg(test)]
    pub fn new_with_max_size_and_step(max_size: i32, step: i32) -> Self {
        Self {
            max_size: Some(max_size),
            step: step.max(1),
            ..Self::default()
        }
    }

    /// Truncates the cache to `len` tokens.
    pub fn truncate(&mut self, len: i32, stream: &Stream) -> Result<(), Exception> {
        if self.attention_window.is_some() {
            if len < 0 || len > self.offset {
                return Err(Exception::custom(format!(
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
                return Err(Exception::custom(format!(
                    "cannot roll a sliding cache back to absolute position {len}; retained state starts at {retained_origin}"
                )));
            }
            let retained = len - retained_origin;
            if let Some(keys) = self.keys.take() {
                self.keys = Some(keys.try_index_device((.., .., ..retained, ..), stream)?);
            }
            if let Some(values) = self.values.take() {
                self.values = Some(values.try_index_device((.., .., ..retained, ..), stream)?);
            }
            self.key_buffer = self
                .key_buffer
                .as_ref()
                .map(|b| b.truncated(retained))
                .transpose()?;
            self.value_buffer = self
                .value_buffer
                .as_ref()
                .map(|b| b.truncated(retained))
                .transpose()?;
            self.offset = len;
            self.length = retained;
            if self.key_buffer.is_none() {
                self.capacity = retained;
            }
            return Ok(());
        }
        if len < 0 || len > self.length {
            return Err(Exception::custom(format!(
                "concatenating cache truncate length {len} is outside 0..{}",
                self.length
            )));
        }
        if self.append_only {
            let key_buffer = self
                .key_buffer
                .as_ref()
                .map(|b| b.truncated(len))
                .transpose()?;
            let value_buffer = self
                .value_buffer
                .as_ref()
                .map(|b| b.truncated(len))
                .transpose()?;
            let keys = self
                .keys
                .as_ref()
                .map(|a| a.try_index_device((.., .., ..len, ..), stream))
                .transpose()?;
            let values = self
                .values
                .as_ref()
                .map(|a| a.try_index_device((.., .., ..len, ..), stream))
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
            self.keys = Some(keys.try_index_device((.., .., ..len, ..), stream)?);
        }
        if let Some(values) = self.values.take() {
            self.values = Some(values.try_index_device((.., .., ..len, ..), stream)?);
        }
        self.offset = len;
        self.length = len;
        self.capacity = len;
        Ok(())
    }

    /// Returns the arrays currently retained by the cache.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
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

impl ConcatKeyValueCache {
    fn grown_capacity(&self, required: i32) -> i32 {
        let step = self.step.max(1);
        let chunks = (required + step - 1) / step;
        let capacity = chunks * step;
        self.max_size
            .map_or(capacity, |max_size| capacity.min(max_size))
    }

    fn update_append_only(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let key_shape = keys.shape();
        let value_shape = values.shape();
        if key_shape.len() != 4 || value_shape.len() != 4 || key_shape[..3] != value_shape[..3] {
            return Err(Exception::custom(
                "cache append requires matching batch/head/token geometry",
            ));
        }
        let required = self
            .length
            .checked_add(key_shape[2])
            .ok_or_else(|| Exception::custom("cache append overflow"))?;
        let capacity = self
            .continuation_capacity_bound(key_shape[2] as u64)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| Exception::custom("cache capacity overflow"))?;
        let candidate =
            |buffer: &Option<PrefixAppendBuffer>, previous: &Option<Array>, update: &Array| {
                if let Some(buffer) = buffer {
                    buffer.appended(update, capacity, stream)
                } else if let Some(previous) = previous {
                    PrefixAppendBuffer::new(previous, capacity, -2, stream)?
                        .appended(update, capacity, stream)
                } else {
                    PrefixAppendBuffer::new(update, capacity, -2, stream)
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
        let keys = key_buffer.prefix(stream)?;
        let values = match &value_buffer {
            Some(buffer) => buffer.prefix(stream)?,
            None => keys.clone(),
        };
        self.capacity = key_buffer.capacity();
        self.key_buffer = Some(key_buffer);
        self.value_buffer = value_buffer;
        self.keys = Some(keys.clone());
        self.values = (!self.key_only).then_some(values.clone());
        self.length = required;
        self.offset = required;
        Ok((keys, values))
    }

    fn padded(array: &Array, capacity: i32, stream: &Stream) -> Result<Array, Exception> {
        let mut shape = array.shape().to_vec();
        let sequence_axis = shape.len() - 2;
        shape[sequence_axis] = capacity;
        zeros_dtype(&shape, array.dtype(), stream)
    }

    fn logical_arrays(&self, stream: &Stream) -> Result<(Array, Array), Exception> {
        let keys = self
            .keys
            .as_ref()
            .expect("Keys cannot be None")
            .try_index_device((.., .., ..self.length, ..), stream)?;
        let values = if self.key_only {
            keys.clone()
        } else {
            self.values
                .as_ref()
                .expect("Values cannot be None")
                .try_index_device((.., .., ..self.length, ..), stream)?
        };
        Ok((keys, values))
    }

    pub fn snapshot_arrays(&self, stream: &Stream) -> Result<Option<(Array, Array)>, Exception> {
        if self.keys.is_none() {
            return Ok(None);
        }
        self.logical_arrays(stream).map(Some)
    }

    pub fn restore_resident(
        &mut self,
        keys: Array,
        values: Array,
        end: i32,
    ) -> Result<(), Exception> {
        let length = keys.dim(-2);
        let compatible = keys.ndim() == values.ndim()
            && keys.ndim() >= 2
            && keys.shape()[..keys.ndim() - 2] == values.shape()[..values.ndim() - 2]
            && values.dim(-2) == length;
        if !compatible || end < length {
            return Err(Exception::custom(
                "restored key/value cache arrays do not match their retained range",
            ));
        }
        if self.attention_window.is_none() && end != length {
            return Err(Exception::custom(
                "restored full-attention cache must begin at token zero",
            ));
        }
        if let Some(window) = self.attention_window {
            if length > window.saturating_sub(1) {
                return Err(Exception::custom(
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

impl KeyValueCache for ConcatKeyValueCache {
    fn memory_retained_positions(&self) -> Option<u64> {
        u64::try_from(self.length).ok()
    }

    fn update_memory_contract(
        &self,
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, ComputeError> {
        use eredu_nn::mechanism_memory::*;
        let MechanismInvocation::CacheUpdate {
            batch,
            heads,
            previous,
            appended,
            key_width,
            value_width,
            element,
        } = *invocation
        else {
            return Err(ComputeError::backend(
                "cache update memory requires a cache append invocation",
            ));
        };
        if previous != u64::try_from(self.length).map_err(ComputeError::backend)? {
            return Err(ComputeError::backend(
                "cache memory append retained frontier differs from installed state",
            ));
        }
        if self.key_only && key_width != value_width {
            return Err(ComputeError::backend(
                "key-only cache value geometry must match keys",
            ));
        }
        for (array, width) in [
            (self.keys.as_ref(), key_width),
            (self.values.as_ref(), value_width),
        ] {
            if let Some(array) = array {
                let shape = array.shape();
                if shape.len() != 4
                    || shape[0] as u64 != batch
                    || shape[1] as u64 != heads
                    || shape[3] as u64 != width
                    || crate::backend::nn::memory::element_type(array.dtype()) != Some(element)
                {
                    return Err(ComputeError::backend(
                        "cache memory append shape/dtype differs from installed arrays",
                    ));
                }
            }
        }
        let mut result = crate::backend::nn::memory::describe(invocation)?;
        result
            .storage
            .retain(|storage| storage.name == "native_workspace");
        result
            .missing
            .retain(|reason| !reason.starts_with("cache append backing"));
        let visible = previous
            .checked_add(appended)
            .ok_or_else(|| ComputeError::backend("cache append overflowed"))?;
        let padded = self
            .continuation_capacity_bound(appended)
            .ok_or_else(|| ComputeError::backend("cache append capacity overflowed"))?;
        // Sliding retention is a view into the complete concatenation. Its
        // shorter logical history does not reduce this backing payload.
        let backing_positions =
            if self.attention_window.is_some() && !self.append_eligible(appended) {
                visible
            } else {
                padded.max(visible)
            };
        for (name, width) in [("cache_keys", key_width), ("cache_values", value_width)] {
            if self.key_only && name == "cache_values" {
                continue;
            }
            let payload = crate::backend::nn::memory::bytes(
                &[batch, heads, backing_positions, width],
                element_bytes(element),
            )?;
            let mut storage = crate::backend::nn::memory::allocation(
                name,
                payload,
                MechanismStorageRole::State,
                StorageRetention::Returned,
            );
            if self.keys.is_none() && backing_positions == appended {
                storage.backing = MechanismBacking::Unknown;
                storage.detail =
                    "empty cache borrows appended tensor backing; no independent cache allocation"
                        .into();
            } else if self.append_eligible(appended)
                && self.key_buffer.as_ref().is_some_and(|b| {
                    i32::try_from(appended).is_ok_and(|n| b.can_append_without_copy(n))
                })
                && (self.key_only
                    || self.value_buffer.as_ref().is_some_and(|b| {
                        i32::try_from(appended).is_ok_and(|n| b.can_append_without_copy(n))
                    }))
            {
                storage.backing = MechanismBacking::Owner(name.into());
                storage.detail = "append reserves an unused suffix of private owner capacity; published prefixes remain immutable".into();
            } else if !self.append_only
                && self.step > 1
                && self.attention_window.is_none()
                && visible <= self.capacity as u64
            {
                storage.backing = MechanismBacking::Owner(name.into());
                storage.detail =
                    "append fits actual reserved cache capacity; mutation reuses owner backing"
                        .into();
            } else {
                storage.detail = "selected cache append allocates concatenated/padded backing; old storage remains a lazy graph dependency".into();
            }
            result.storage.push(storage);
        }
        result.missing.push("cache input layouts and native update copy-on-write/old-buffer retention require completion/lifetime composition".into());
        result.validate()?;
        Ok(result)
    }

    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        self.attention_window.or(self.max_size)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        self.keys.iter().chain(self.values.iter()).collect()
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let new_tokens = keys.dim(-2);
        if self.append_eligible(new_tokens as u64) {
            return self.update_append_only(keys, values, stream);
        }
        self.offset += new_tokens;

        if let Some(window) = self.attention_window {
            // Cropping changes the prefix origin. No capacity owner may reserve
            // suffixes relative to the old origin after retention begins.
            self.key_buffer = None;
            self.value_buffer = None;
            let combined_keys = match self.keys.take() {
                Some(previous) => append_immutable(previous, keys, "cache_append_keys", stream)?,
                None => keys,
            };
            let combined_values = if self.key_only {
                combined_keys.clone()
            } else {
                match self.values.take() {
                    Some(previous) => {
                        append_immutable(previous, values, "cache_append_values", stream)?
                    }
                    None => values,
                }
            };
            let retained = (window - 1).min(combined_keys.dim(-2));
            if retained == 0 {
                self.keys = None;
                self.values = None;
            } else {
                let start = combined_keys.dim(-2) - retained;
                self.keys = Some(combined_keys.try_index_device((.., .., start.., ..), stream)?);
                self.values = if self.key_only {
                    None
                } else {
                    Some(combined_values.try_index_device((.., .., start.., ..), stream)?)
                };
            }
            self.length = retained;
            self.capacity = retained;
            return Ok((combined_keys, combined_values));
        }

        if self.step <= 1 {
            if self.key_only {
                self.keys = Some(match self.keys.take() {
                    Some(previous) => {
                        append_immutable(previous, keys, "cache_append_keys", stream)?
                    }
                    None => keys,
                });
            } else {
                match (self.keys.take(), self.values.take()) {
                    (Some(k), Some(v)) => {
                        self.keys = Some(append_immutable(k, keys, "cache_append_keys", stream)?);
                        self.values =
                            Some(append_immutable(v, values, "cache_append_values", stream)?);
                    }
                    _ => {
                        self.keys = Some(keys);
                        self.values = Some(values);
                    }
                }
            }
            if let Some(max_size) = self.max_size {
                let length = self.keys.as_ref().expect("Keys cannot be None").dim(-2);
                if length > max_size {
                    let start = length - max_size;
                    self.keys = Some(
                        self.keys
                            .take()
                            .expect("Keys cannot be None")
                            .try_index_device((.., .., start.., ..), stream)?,
                    );
                    if !self.key_only {
                        self.values = Some(
                            self.values
                                .take()
                                .expect("Values cannot be None")
                                .try_index_device((.., .., start.., ..), stream)?,
                        );
                    }
                }
            }
            self.length = self.keys.as_ref().expect("Keys cannot be None").dim(-2);
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

        let required = self.length + new_tokens;

        if let Some(max_size) = self.max_size {
            if required > max_size {
                if self.keys.is_none() {
                    let start = new_tokens - max_size;
                    self.keys = Some(keys.try_index_device((.., .., start.., ..), stream)?);
                    if !self.key_only {
                        self.values = Some(values.try_index_device((.., .., start.., ..), stream)?);
                    }
                    self.length = max_size;
                    self.capacity = max_size;
                    return self.logical_arrays(stream);
                }
                let (old_keys, old_values) = self.logical_arrays(stream)?;
                let combined_keys = concatenate_axis(&[old_keys, keys], -2, stream)?;
                let combined_values = concatenate_axis(&[old_values, values], -2, stream)?;
                let start = required - max_size;
                self.keys = Some(combined_keys.try_index_device((.., .., start.., ..), stream)?);
                if !self.key_only {
                    self.values =
                        Some(combined_values.try_index_device((.., .., start.., ..), stream)?);
                }
                self.length = max_size;
                self.capacity = max_size;
                return self.logical_arrays(stream);
            }
        }

        if self.keys.is_none() {
            self.capacity = self.grown_capacity(required);
            if self.capacity == required {
                self.keys = Some(keys);
                if !self.key_only {
                    self.values = Some(values);
                }
                self.length = required;
                return self.logical_arrays(stream);
            }
            self.keys = Some(Self::padded(&keys, self.capacity, stream)?);
            if !self.key_only {
                self.values = Some(Self::padded(&values, self.capacity, stream)?);
            }
        } else if required > self.capacity {
            let new_capacity = self.grown_capacity(required);
            let padding = new_capacity - self.capacity;
            let key_padding = Self::padded(&keys, padding, stream)?;
            let value_padding = (!self.key_only)
                .then(|| Self::padded(&values, padding, stream))
                .transpose()?;
            self.keys = Some(concatenate_axis(
                &[self.keys.take().expect("Keys cannot be None"), key_padding],
                -2,
                stream,
            )?);
            if let Some(value_padding) = value_padding {
                self.values = Some(concatenate_axis(
                    &[
                        self.values.take().expect("Values cannot be None"),
                        value_padding,
                    ],
                    -2,
                    stream,
                )?);
            }
            self.capacity = new_capacity;
        }

        self.keys
            .as_mut()
            .expect("Keys cannot be None")
            .try_index_mut_device((.., .., self.length..required, ..), &keys, stream)?;
        if !self.key_only {
            self.values
                .as_mut()
                .expect("Values cannot be None")
                .try_index_mut_device((.., .., self.length..required, ..), &values, stream)?;
        }
        self.length = required;
        self.logical_arrays(stream)
    }
}

impl eredu_runtime::RuntimeLayerState<MlxNeuralBackend> for ConcatKeyValueCache {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        [
            self.keys.as_ref().map(retained_tensor),
            self.values.as_ref().map(retained_tensor),
        ]
        .into_iter()
        .flatten()
    }
}

#[cfg(test)]
mod mechanism_memory_tests {
    use super::*;
    use eredu_nn::{mechanism_memory::*, TensorElementType};
    fn request(previous: u64, appended: u64) -> MechanismInvocation {
        MechanismInvocation::CacheUpdate {
            batch: 1,
            heads: 2,
            previous,
            appended,
            key_width: 3,
            value_width: 3,
            element: TensorElementType::F32,
        }
    }
    fn input(tokens: i32) -> Array {
        Array::from_slice(&vec![0.5f32; (2 * tokens * 3) as usize], &[1, 2, tokens, 3])
    }
    #[test]
    fn mechanism_cache_descriptions_preserve_frontier_and_backing() {
        let stream =
            safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut cache = ConcatKeyValueCache::new_with_step(8);
        let empty = KeyValueCache::update_memory_contract(&cache, &request(0, 3)).unwrap();
        assert_eq!(cache.offset, 0);
        assert_eq!(
            empty
                .storage
                .iter()
                .find(|entry| entry.name == "cache_keys")
                .unwrap()
                .payload,
            MechanismBytes::exact(2 * 8 * 3 * 4)
        );
        cache.update_and_fetch(input(3), input(3), &stream).unwrap();
        let before = (
            cache.offset,
            cache.length,
            cache.capacity,
            cache.keys.as_ref().unwrap().graph_identity(),
        );
        let contract = KeyValueCache::update_memory_contract(&cache, &request(3, 2)).unwrap();
        assert_eq!(
            contract,
            KeyValueCache::update_memory_contract(&cache, &request(3, 2)).unwrap()
        );
        assert_eq!(
            before,
            (
                cache.offset,
                cache.length,
                cache.capacity,
                cache.keys.as_ref().unwrap().graph_identity()
            )
        );
        let keys = contract
            .storage
            .iter()
            .find(|entry| entry.name == "cache_keys")
            .unwrap();
        assert_eq!(keys.backing, MechanismBacking::Owner("cache_keys".into()));
        assert_eq!(keys.payload, MechanismBytes::exact(2 * 8 * 3 * 4));
        let growth = KeyValueCache::update_memory_contract(&cache, &request(3, 9)).unwrap();
        let keys = growth
            .storage
            .iter()
            .find(|entry| entry.name == "cache_keys")
            .unwrap();
        assert_eq!(keys.payload, MechanismBytes::exact(2 * 16 * 3 * 4));
        assert_eq!(keys.backing, MechanismBacking::Invocation);
        assert!(KeyValueCache::update_memory_contract(&cache, &request(2, 1)).is_err());
        let mut mismatched = request(3, 1);
        if let MechanismInvocation::CacheUpdate { heads, .. } = &mut mismatched {
            *heads = 1;
        }
        assert!(KeyValueCache::update_memory_contract(&cache, &mismatched).is_err());
    }
    #[test]
    fn mechanism_sliding_cache_prices_complete_concatenation_and_key_only_sharing() {
        let stream =
            safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut cache = ConcatKeyValueCache::new_key_only_for_sliding_attention(4);
        cache.update_and_fetch(input(6), input(6), &stream).unwrap();
        assert_eq!((cache.offset, cache.length), (6, 3));
        let contract = KeyValueCache::update_memory_contract(&cache, &request(3, 8)).unwrap();
        assert!(!contract
            .storage
            .iter()
            .any(|entry| entry.name == "cache_values"));
        assert_eq!(
            contract
                .storage
                .iter()
                .find(|entry| entry.name == "cache_keys")
                .unwrap()
                .payload,
            MechanismBytes::exact(2 * 11 * 3 * 4)
        );
        assert_eq!((cache.offset, cache.length), (6, 3));
    }
}

// Immutable append roots can be observed without copying mutable backing storage.
fn append_immutable(
    previous: Array,
    next: Array,
    _role: &'static str,
    stream: &Stream,
) -> Result<Array, Exception> {
    let output = concatenate_axis(&[&previous, &next], -2, stream)?;
    #[cfg(feature = "projection-profiling")]
    {
        use crate::backend::nn::projection_profile::operations::{record, Operation};
        record(
            Operation::Concatenate(-2),
            _role,
            &previous,
            &next,
            &output,
            stream,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod append_only_tests {
    use super::*;
    use eredu_nn::{mechanism_memory::*, TensorElementType};

    #[test]
    fn append_candidates_preserve_roots_and_report_private_capacity() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let input = |x: f32| Array::from_slice(&[x, x + 1.], &[2, 1, 1, 1]);
        let read = |a: Array| a.into_evaluated().unwrap().try_to_vec::<f32>().unwrap();
        let mut owner = ConcatKeyValueCache::new_append_only();
        owner
            .update_and_fetch(input(1.), input(-2.), &stream)
            .unwrap();
        let request = MechanismInvocation::CacheUpdate {
            batch: 2,
            heads: 1,
            previous: 1,
            appended: 1,
            key_width: 1,
            value_width: 1,
            element: TensorElementType::F32,
        };
        assert!(owner.has_immutable_roots());
        assert_eq!(owner.spare_capacity_bytes(), Some(2 * 127 * 2 * 4));
        let storage = owner.update_memory_contract(&request).unwrap();
        assert!(storage
            .storage
            .iter()
            .filter(|s| s.name.starts_with("cache_"))
            .all(|s| {
                s.payload == MechanismBytes::exact(2 * 128 * 4)
                    && matches!(s.backing, MechanismBacking::Owner(_))
            }));
        let original = owner.snapshot_arrays(&stream).unwrap().unwrap();
        let snapshot = owner.isolated_snapshot(&stream).unwrap();
        let mut discarded = owner.checkpoint_clone_state().unwrap();
        let lazy = discarded
            .update_and_fetch(input(3.), input(-4.), &stream)
            .unwrap();
        assert!(owner
            .update_memory_contract(&request)
            .unwrap()
            .storage
            .iter()
            .filter(|s| s.name.starts_with("cache_"))
            .all(|s| !matches!(s.backing, MechanismBacking::Owner(_))));
        let mut committed = owner.checkpoint_clone_state().unwrap();
        let selected = committed
            .update_and_fetch(input(5.), input(-6.), &stream)
            .unwrap();
        assert_eq!(read(selected.0), [1., 5., 2., 6.]);
        assert_eq!(read(lazy.0), [1., 3., 2., 4.]);
        assert_eq!(read(original.0), [1., 2.]);
        assert_eq!(
            read(snapshot.snapshot_arrays(&stream).unwrap().unwrap().1),
            [-2., -1.]
        );
        // Invalid second updates consume no visible publication, even when a
        // key reservation is made before the value dtype error is detected.
        let before = committed.offset();
        let wrong = Array::from_slice(&[1i32, 2], &[2, 1, 1, 1]);
        assert!(committed
            .update_and_fetch(input(7.), wrong, &stream)
            .is_err());
        assert_eq!(committed.offset(), before);
        committed.truncate(1, &stream).unwrap();
        assert_eq!(
            read(
                committed
                    .update_and_fetch(input(9.), input(-10.), &stream)
                    .unwrap()
                    .0
            ),
            [1., 9., 2., 10.]
        );
        assert_eq!(
            read(discarded.snapshot_arrays(&stream).unwrap().unwrap().0),
            [1., 3., 2., 4.]
        );
    }
}

#[cfg(test)]
mod sliding_append_tests {
    use super::*;
    #[test]
    fn reserved_sliding_prefixes_match_concat_through_truncate_and_crop() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        for key_only in [false, true] {
            let mut optimized = if key_only {
                ConcatKeyValueCache::new_key_only_append_only_for_sliding_attention(4)
            } else {
                ConcatKeyValueCache::new_append_only_for_sliding_attention(4)
            };
            let mut reference = if key_only {
                ConcatKeyValueCache::new_key_only_for_sliding_attention(4)
            } else {
                ConcatKeyValueCache::new_for_sliding_attention(4)
            };
            let input =
                |values: &[f32]| Array::from_slice(values, &[2, 1, values.len() as i32 / 2, 1]);
            let read = |a: Array| a.into_evaluated().unwrap().try_to_vec::<f32>().unwrap();
            for (round, data) in [
                vec![1., 2., 11., 12.],
                vec![3., 13.],
                vec![4., 5., 14., 15.],
                vec![6., 16.],
            ]
            .into_iter()
            .enumerate()
            {
                let a = optimized
                    .update_and_fetch(input(&data), input(&data), &stream)
                    .unwrap();
                let b = reference
                    .update_and_fetch(input(&data), input(&data), &stream)
                    .unwrap();
                assert_eq!(read(a.0), read(b.0));
                assert_eq!(read(a.1), read(b.1));
                assert_eq!(optimized.offset(), reference.offset());
                if round == 0 {
                    assert!(optimized.key_buffer.is_some());
                    assert_eq!(optimized.continuation_capacity_bound(1), Some(3));
                    optimized.truncate(1, &stream).unwrap();
                    reference.truncate(1, &stream).unwrap();
                }
                if round >= 2 {
                    assert!(optimized.key_buffer.is_none());
                }
                let a = optimized.snapshot_arrays(&stream).unwrap().unwrap();
                let b = reference.snapshot_arrays(&stream).unwrap().unwrap();
                assert_eq!(read(a.0), read(b.0));
                assert_eq!(read(a.1), read(b.1));
            }
            let origin = optimized.offset() - optimized.length;
            optimized.truncate(origin + 1, &stream).unwrap();
            reference.truncate(origin + 1, &stream).unwrap();
            let a = optimized
                .update_and_fetch(input(&[7., 17.]), input(&[7., 17.]), &stream)
                .unwrap();
            let b = reference
                .update_and_fetch(input(&[7., 17.]), input(&[7., 17.]), &stream)
                .unwrap();
            assert_eq!(read(a.0), read(b.0));
            assert_eq!(optimized.offset(), reference.offset());
        }
    }
}
