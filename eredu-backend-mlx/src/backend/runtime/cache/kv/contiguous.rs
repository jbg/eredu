//! Contiguous append-only key/value cache storage.

use super::*;

/// A cache that appends all key/value states along the sequence axis.
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    key_only: bool,
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
    /// Maximum logical sequence dimension retained after the admitted input
    /// span. Includes chunk padding and existing capacity; sliding views are
    /// priced by their logical extent, not the allocator's underlying buffer.
    pub(crate) fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        let required = u64::try_from(self.length).ok()?.checked_add(additional)?;
        let bound = if let Some(window) = self.attention_window {
            required.min(u64::try_from(window.checked_sub(1)?).ok()?)
        } else {
            let step = u64::try_from(self.step.max(1)).ok()?;
            let padded = required
                .checked_add(step - 1)?
                .checked_div(step)?
                .checked_mul(step)?;
            match self.max_size {
                Some(maximum) => padded.min(u64::try_from(maximum).ok()?),
                None => padded,
            }
        };
        Some(bound.max(u64::try_from(self.capacity).ok()?))
    }

    /// Independent snapshot of every logical element, including strided/windowed
    /// views. Preserve capacity, offsets and window metadata with the copied arrays.
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
        Ok(cache)
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
        Ok(cache)
    }

    /// Snapshots the ordinary append-only cache without changing array layout.
    /// Capacity-backed caches can update storage in place and therefore still
    /// require an independent data copy.
    pub fn checkpoint_clone_state(&self) -> Result<Self, Exception> {
        if self.step <= 1 {
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
            self.offset = len;
            self.length = retained;
            self.capacity = retained;
            return Ok(());
        }
        if len < 0 || len > self.length {
            return Err(Exception::custom(format!(
                "concatenating cache truncate length {len} is outside 0..{}",
                self.length
            )));
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
        self.offset = end;
        self.length = length;
        self.capacity = length;
        Ok(())
    }
}

impl KeyValueCache for ConcatKeyValueCache {
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
        self.offset += new_tokens;

        if let Some(window) = self.attention_window {
            let combined_keys = match self.keys.take() {
                Some(previous) => concatenate_axis(&[previous, keys], -2, stream)?,
                None => keys,
            };
            let combined_values = if self.key_only {
                combined_keys.clone()
            } else {
                match self.values.take() {
                    Some(previous) => concatenate_axis(&[previous, values], -2, stream)?,
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
                    Some(previous) => concatenate_axis(&[previous, keys], -2, stream)?,
                    None => keys,
                });
            } else {
                match (self.keys.take(), self.values.take()) {
                    (Some(k), Some(v)) => {
                        self.keys = Some(concatenate_axis(&[k, keys], -2, stream)?);
                        self.values = Some(concatenate_axis(&[v, values], -2, stream)?);
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
        self.keys
            .iter()
            .chain(self.values.iter())
            .map(retained_tensor)
    }
}
