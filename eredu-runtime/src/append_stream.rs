//! Named append-only tensor streams with bounded range access.
use eredu_core::cache::append_stream::{
    AppendOnlyStream, AppendStreamBacking, AppendStreamBinding, AppendStreamError,
    AppendStreamLimits, AppendStreamSpec,
};
use eredu_nn::{Index, Tensor, TensorElementType};
use std::ops::Range;

/// Checks selected scalar types as well as complete logical layout coverage.
pub fn resolve_append_stream_allocations(
    selected: &crate::SelectedStateRealization,
    bindings: &[AppendStreamBinding],
) -> Result<Vec<(AppendStreamBinding, AppendStreamBacking)>, AppendStreamError> {
    validate_append_stream_layout(selected.layout(), bindings)?;
    let mut allocations = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let component = selected
            .components()
            .iter()
            .find(|c| {
                c.layer() == binding.layer
                    && c.component().role()
                        == eredu_core::cache::StateComponentRole::AppendStream {
                            slot: binding.spec.slot,
                        }
            })
            .ok_or(AppendStreamError::Geometry)?;
        if component.storage_dtype().element_type() != binding.spec.element {
            return Err(AppendStreamError::Geometry);
        }
        let backing = match component.placement() {
            crate::StateComponentPlacement::Device => {
                let required = (binding.limits.entries as u64)
                    .checked_mul(binding.spec.record_bytes()?)
                    .ok_or(AppendStreamError::Geometry)?;
                if required > binding.payload_bytes {
                    return Err(AppendStreamError::Budget {
                        resource: "payload bytes",
                        required,
                        limit: binding.payload_bytes,
                    });
                }
                AppendStreamBacking::Resident
            }
            crate::StateComponentPlacement::Paged => {
                let crate::CacheResidencyPolicy::Paged(options) = selected.policy() else {
                    return Err(AppendStreamError::Geometry);
                };
                if binding.limits.page_entries != options.block_size_tokens() as usize {
                    return Err(AppendStreamError::Geometry);
                }
                let required = (binding.limits.page_entries as u64)
                    .checked_mul(binding.spec.record_bytes()?)
                    .ok_or(AppendStreamError::Geometry)?;
                if required > options.device_budget_bytes() {
                    return Err(AppendStreamError::Budget {
                        resource: "device page bytes",
                        required,
                        limit: options.device_budget_bytes(),
                    });
                }
                AppendStreamBacking::Paged
            }
        };
        allocations.push((binding.clone(), backing));
    }
    Ok(allocations)
}
/// Proves complete, unique coverage of all declared streams before allocating.
pub fn validate_append_stream_layout<'a>(
    layout: &crate::StateLayout,
    bindings: impl IntoIterator<IntoIter: ExactSizeIterator, Item = &'a AppendStreamBinding>,
) -> Result<(), AppendStreamError> {
    let bindings = bindings.into_iter();
    let declared = layout
        .layers()
        .iter()
        .map(|p| p.append_streams().len())
        .sum::<usize>();
    if declared != bindings.len() {
        return Err(AppendStreamError::Geometry);
    }
    let mut allowances = AppendStreamAllowances::default();
    let mut identities = std::collections::BTreeSet::new();
    let mut lanes = None;
    for binding in bindings {
        if binding.lanes == 0
            || binding.lanes > i32::MAX as u32
            || lanes.is_some_and(|lanes| binding.lanes != lanes)
            || !identities.insert((binding.layer, binding.spec.slot))
        {
            return Err(AppendStreamError::Geometry);
        }
        lanes = Some(binding.lanes);
        allowances.add(binding)?;
        let policy = layout
            .layer(binding.layer)
            .and_then(|policy| {
                policy
                    .append_streams()
                    .iter()
                    .find(|s| s.slot() == binding.spec.slot)
            })
            .ok_or(AppendStreamError::Geometry)?;
        let dtype = match binding.spec.element {
            TensorElementType::F16 => "Float16",
            TensorElementType::Bf16 => "Bfloat16",
            TensorElementType::F32 => "Float32",
            TensorElementType::F64 => "Float64",
            TensorElementType::I32 => "Int32",
            TensorElementType::U32 => "Uint32",
            _ => return Err(AppendStreamError::Geometry),
        };
        if binding.spec.width != policy.width() || !policy.dtype().accepts_dtype_name(dtype) {
            return Err(AppendStreamError::Geometry);
        }
        let required = binding.limits.scratch_bytes(&binding.spec)?;
        if required > binding.scratch_bytes {
            return Err(AppendStreamError::Budget {
                resource: "scratch bytes",
                required,
                limit: binding.scratch_bytes,
            });
        }
    }
    Ok(())
}

/// Adds declared per-lane allowances without hiding replicas or overflowing.
/// Paged payload residency is charged by the shared cache policy separately.
pub fn append_stream_allowances<'a>(
    bindings: impl IntoIterator<Item = &'a AppendStreamBinding>,
) -> Result<AppendStreamAllowances, AppendStreamError> {
    let mut total = AppendStreamAllowances::default();
    for binding in bindings {
        total.add(binding)?;
    }
    Ok(total)
}

impl AppendStreamAllowances {
    fn add(&mut self, binding: &AppendStreamBinding) -> Result<(), AppendStreamError> {
        for (value, limit) in [
            (&mut self.payload_bytes, binding.payload_bytes),
            (&mut self.scratch_bytes, binding.scratch_bytes),
            (&mut self.catalog_bytes, binding.catalog_bytes),
        ] {
            *value = limit
                .checked_mul(u64::from(binding.lanes))
                .and_then(|n| value.checked_add(n))
                .ok_or(AppendStreamError::Geometry)?;
        }
        Ok(())
    }
}

/// Aggregate declared allowances across all local streams and lanes.
/// These are limits, not observations of allocated bytes or shared paged residency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AppendStreamAllowances {
    /// Sum of lane-local resident payload allowances.
    pub payload_bytes: u64,
    /// Sum of lane-local tail and read workspace allowances.
    pub scratch_bytes: u64,
    /// Sum of lane-local host catalog allowances.
    pub catalog_bytes: u64,
}

/// Named lane-local stream access for an architecture's combined layer state.
pub trait RuntimeAppendStreams<B: eredu_nn::NeuralBackend>: crate::RuntimeLayerState<B> {
    /// Resident or paged stream type selected by the retained state plan.
    type Stream: AppendOnlyStream<B::Tensor>;
    /// Borrows exactly one declared stream and lane.
    fn append_stream(
        &mut self,
        slot: u32,
        lane: u32,
    ) -> Result<&mut Self::Stream, AppendStreamError>;
    /// Borrows two distinct slots together, for paired keys and integer positions.
    fn append_stream_pair(
        &mut self,
        first: u32,
        second: u32,
        lane: u32,
    ) -> Result<(&mut Self::Stream, &mut Self::Stream), AppendStreamError>;
}

/// Resident realization shared by neutral conformance and native tensor contexts.
///
/// Completed chunks are immutable. Clones retain an exact cheap transaction
/// checkpoint; appending or reading never modifies a checkpoint's tensor storage.
#[derive(Debug, Clone)]
pub struct ResidentAppendStream<T: Tensor> {
    spec: AppendStreamSpec,
    limits: AppendStreamLimits,
    entries: usize,
    chunks: Vec<T>,
}
impl<T: Tensor> ResidentAppendStream<T> {
    /// Admits total payload, bounded tail/range scratch and catalog metadata before use.
    pub fn new(
        spec: AppendStreamSpec,
        limits: AppendStreamLimits,
        payload_bytes: u64,
        scratch_bytes: u64,
        catalog_bytes: u64,
    ) -> Result<Self, AppendStreamError> {
        let record = spec.record_bytes()?;
        limits.validate()?;
        let payload = (limits.entries as u64)
            .checked_mul(record)
            .ok_or(AppendStreamError::Geometry)?;
        let scratch = limits.scratch_bytes(&spec)?;
        let catalog = (limits.entries.div_ceil(limits.page_entries) as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or(AppendStreamError::Geometry)?;
        for (resource, required, limit) in [
            ("payload bytes", payload, payload_bytes),
            ("scratch bytes", scratch, scratch_bytes),
            ("catalog bytes", catalog, catalog_bytes),
        ] {
            if required > limit {
                return Err(AppendStreamError::Budget {
                    resource,
                    required,
                    limit,
                });
            }
        }
        Ok(Self {
            spec,
            limits,
            entries: 0,
            chunks: Vec::new(),
        })
    }
    /// Tensor roots retained by state completion and observation owners.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter()
    }
    /// Capacity of the retained host chunk-handle catalog, excluding tensor payload.
    pub fn catalog_bytes(&self) -> Option<u64> {
        (self.chunks.capacity() as u64).checked_mul(std::mem::size_of::<T>() as u64)
    }
    /// Copies every retained root with a backend-supplied isolation operation.
    /// Geometry and logical frontiers are preserved; failures leave this state intact.
    pub fn try_clone_with<E>(&self, mut copy: impl FnMut(&T) -> Result<T, E>) -> Result<Self, E> {
        Ok(Self {
            spec: self.spec.clone(),
            limits: self.limits,
            entries: self.entries,
            chunks: self
                .chunks
                .iter()
                .map(&mut copy)
                .collect::<Result<_, _>>()?,
        })
    }
    /// Clears only logical state. Session observation and copy budgets belong to the driver.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.entries = 0;
    }
}
/// Checks an append against the stream geometry, current frontier, and entry budget.
pub fn admit_append(
    spec: &AppendStreamSpec,
    limits: &AppendStreamLimits,
    entries: usize,
    frontier: usize,
    values: &impl Tensor,
) -> Result<usize, AppendStreamError> {
    if frontier != entries {
        return Err(AppendStreamError::Frontier {
            expected: frontier,
            actual: entries,
        });
    }
    if values.shape().len() != 2
        || values.dim(0) <= 0
        || values.dim(1) != spec.width
        || values.element_type() != Some(spec.element)
    {
        return Err(AppendStreamError::Geometry);
    }
    let added = values.dim(0) as usize;
    let next = entries
        .checked_add(added)
        .ok_or(AppendStreamError::Geometry)?;
    if next > limits.entries {
        return Err(AppendStreamError::Budget {
            resource: "entries",
            required: next as u64,
            limit: limits.entries as u64,
        });
    }
    Ok(next)
}

/// Checks that a read lies within the current frontier and read budget.
pub fn admit_read(
    limits: &AppendStreamLimits,
    entries: usize,
    range: &Range<usize>,
) -> Result<(), AppendStreamError> {
    if range.start >= range.end || range.end > entries {
        return Err(AppendStreamError::Range);
    }
    if range.len() > limits.read_entries {
        return Err(AppendStreamError::Budget {
            resource: "read entries",
            required: range.len() as u64,
            limit: limits.read_entries as u64,
        });
    }
    Ok(())
}

impl<T: Tensor> AppendOnlyStream<T> for ResidentAppendStream<T> {
    fn specification(&self) -> &AppendStreamSpec {
        &self.spec
    }
    fn len(&self) -> usize {
        self.entries
    }
    fn append(
        &mut self,
        frontier: usize,
        values: T,
        context: &T::Context,
    ) -> Result<(), AppendStreamError> {
        let next = admit_append(&self.spec, &self.limits, self.entries, frontier, &values)?;
        let added = next - self.entries;
        let mut new_chunks = Vec::new();
        let mut consumed = 0;
        let tail = self.entries % self.limits.page_entries;
        if tail > 0 {
            let count = added.min(self.limits.page_entries - tail);
            let part = values.index(&[Index::Range(0, count as i32), Index::Full], context)?;
            new_chunks.push(
                T::concatenate(
                    &[self.chunks.last().expect("nonempty tail").clone(), part],
                    0,
                    context,
                )?
                .compact(context)?,
            );
            consumed = count;
        }
        while consumed < added {
            let end = (consumed + self.limits.page_entries).min(added);
            new_chunks.push(
                values
                    .index(
                        &[Index::Range(consumed as i32, end as i32), Index::Full],
                        context,
                    )?
                    .compact(context)?,
            );
            consumed = end;
        }
        if tail > 0 {
            self.chunks.pop();
        }
        self.chunks.extend(new_chunks);
        self.entries = next;
        Ok(())
    }
    fn read(&mut self, range: Range<usize>, context: &T::Context) -> Result<T, AppendStreamError> {
        admit_read(&self.limits, self.entries, &range)?;
        let page = self.limits.page_entries;
        let mut parts = Vec::new();
        for index in range.start / page..=(range.end - 1) / page {
            let start = range.start.saturating_sub(index * page);
            let end = (range.end - index * page).min(self.chunks[index].dim(0) as usize);
            parts.push(self.chunks[index].index(
                &[Index::Range(start as i32, end as i32), Index::Full],
                context,
            )?);
        }
        Ok(if parts.len() == 1 {
            parts.remove(0).compact(context)?
        } else {
            T::concatenate(&parts, 0, context)?.compact(context)?
        })
    }
}
