//! Named append-only tensor streams with bounded range access.
use eredu_nn::{Error, Index, Tensor, TensorElementType};
use std::ops::Range;

/// Exact lane count and finite storage limits bound before native state allocation.
/// Layer indices are local to the selected state layout. Byte limits are per lane;
/// admission must account for all lanes and all replicas of a binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendStreamBinding {
    /// Local state layer owning this stream.
    pub layer: usize,
    /// Number of independent sequence lanes.
    pub lanes: u32,
    /// Exact parameter-free record geometry.
    pub spec: AppendStreamSpec,
    /// Finite retained-history and operator-read limits.
    pub limits: AppendStreamLimits,
    /// Resident payload allowance for one lane.
    pub payload_bytes: u64,
    /// Compact read and tail-copy allowance for one lane.
    pub scratch_bytes: u64,
    /// Resident chunk catalog allowance for one lane.
    pub catalog_bytes: u64,
}

impl AppendStreamBinding {
    /// Checks selected scalar types as well as complete logical layout coverage.
    pub fn validate_selected(
        selected: &crate::SelectedStateRealization,
        bindings: &[Self],
    ) -> Result<(), AppendStreamError> {
        Self::validate_layout(selected.layout(), bindings)?;
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
            match component.placement() {
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
                }
            }
        }
        Ok(())
    }
    /// Proves complete, unique coverage of all declared streams before allocating.
    pub fn validate_layout(
        layout: &crate::StateLayout,
        bindings: &[Self],
    ) -> Result<(), AppendStreamError> {
        let declared = layout
            .layers()
            .iter()
            .map(|p| p.append_streams().len())
            .sum::<usize>();
        if declared != bindings.len() {
            return Err(AppendStreamError::Geometry);
        }
        let mut identities = std::collections::BTreeSet::new();
        let lanes = bindings.first().map(|binding| binding.lanes);
        for binding in bindings {
            if binding.lanes == 0
                || binding.lanes > i32::MAX as u32
                || Some(binding.lanes) != lanes
                || !identities.insert((binding.layer, binding.spec.slot))
            {
                return Err(AppendStreamError::Geometry);
            }
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
        Self::allowances(bindings)?;
        Ok(())
    }

    /// Adds declared per-lane allowances without hiding replicas or overflowing.
    /// Paged payload residency is charged by the shared cache policy separately.
    pub fn allowances(bindings: &[Self]) -> Result<AppendStreamAllowances, AppendStreamError> {
        let mut total = AppendStreamAllowances::default();
        for binding in bindings {
            for (value, limit) in [
                (&mut total.payload_bytes, binding.payload_bytes),
                (&mut total.scratch_bytes, binding.scratch_bytes),
                (&mut total.catalog_bytes, binding.catalog_bytes),
            ] {
                *value = limit
                    .checked_mul(u64::from(binding.lanes))
                    .and_then(|n| value.checked_add(n))
                    .ok_or(AppendStreamError::Geometry)?;
            }
        }
        Ok(total)
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

/// Exact identity, scalar representation and record width declared by an architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendStreamSpec {
    /// Stable slot within a layer and sequence lane.
    pub slot: u32,
    /// Number of scalar fields in one immutable record.
    pub width: i32,
    /// Exact storage type; integer columns are never promoted through floating point.
    pub element: TensorElementType,
}
impl AppendStreamSpec {
    /// Validates geometry and returns exact logical storage bytes per record.
    pub fn record_bytes(&self) -> Result<u64, AppendStreamError> {
        let bytes = match self.element {
            TensorElementType::Bool | TensorElementType::I8 | TensorElementType::U8 => 1,
            TensorElementType::F16
            | TensorElementType::Bf16
            | TensorElementType::I16
            | TensorElementType::U16 => 2,
            TensorElementType::F32 | TensorElementType::I32 | TensorElementType::U32 => 4,
            TensorElementType::F64
            | TensorElementType::I64
            | TensorElementType::U64
            | TensorElementType::Complex64 => 8,
        };
        if self.width <= 0 {
            return Err(AppendStreamError::Geometry);
        }
        Ok(self.width as u64 * bytes)
    }
}
/// Mechanism limits selected independently of stream identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendStreamLimits {
    /// Maximum immutable record count retained by this mechanism.
    pub entries: usize,
    /// Entries per physical chunk, including the mutable tail.
    pub page_entries: usize,
    /// Largest compact range returned to an operator.
    pub read_entries: usize,
}
impl AppendStreamLimits {
    /// Checks addressable, nonzero logical and physical bounds.
    pub fn validate(&self) -> Result<(), AppendStreamError> {
        if [self.entries, self.page_entries, self.read_entries]
            .into_iter()
            .any(|n| n == 0 || n > i32::MAX as usize)
        {
            return Err(AppendStreamError::Geometry);
        }
        Ok(())
    }

    /// Tail replacement and compact range copies can coexist with their sources.
    pub fn scratch_bytes(&self, spec: &AppendStreamSpec) -> Result<u64, AppendStreamError> {
        self.validate()?;
        (self.page_entries as u64 + self.read_entries as u64)
            .checked_mul(spec.record_bytes()?)
            .and_then(|n| n.checked_mul(2))
            .ok_or(AppendStreamError::Geometry)
    }
}
/// Typed admission, frontier and payload failures.
#[derive(Debug, thiserror::Error)]
pub enum AppendStreamError {
    /// Invalid dimensions, type or chunking policy.
    #[error("append stream geometry or scalar representation differs from its declaration")]
    Geometry,
    /// Exact append frontier did not match retained state.
    #[error("append stream frontier {actual} differs from expected {expected}")]
    Frontier {
        /// Caller frontier.
        expected: usize,
        /// Retained frontier.
        actual: usize,
    },
    /// A requested range is outside retained history.
    #[error("append stream range is outside retained history")]
    Range,
    /// Required storage or range exceeds the selected bound.
    #[error("append stream needs {required} {resource}, admitted {limit}")]
    Budget {
        /// Bounded resource.
        resource: &'static str,
        /// Required amount.
        required: u64,
        /// Admitted amount.
        limit: u64,
    },
    /// Backend range transfer, tensor construction or completion failure.
    #[error(transparent)]
    Tensor(#[from] Error),
}
/// Storage contract used by architecture-owned streaming algorithms.
///
/// Implementations may retain resident chunks or page immutable records through
/// a shared residency manager. Reads return only the requested compact records;
/// they must not gather the complete stream. Append/frontier updates are atomic
/// on synchronous failure. The execution owner retains native work until completion
/// and checkpoints all streams together with ordinary attention and fixed state.
pub trait AppendOnlyStream<T: Tensor> {
    /// Exact declaration retained at construction.
    fn specification(&self) -> &AppendStreamSpec;
    /// Records already committed to this stream.
    fn len(&self) -> usize;
    /// Whether the stream contains no records.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Appends `[entries,width]` at the specified exact frontier.
    fn append(
        &mut self,
        frontier: usize,
        values: T,
        context: &T::Context,
    ) -> Result<(), AppendStreamError>;
    /// Returns compact `[range.len(),width]` storage, within the read bound.
    fn read(&mut self, range: Range<usize>, context: &T::Context) -> Result<T, AppendStreamError>;
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
        if frontier != self.entries {
            return Err(AppendStreamError::Frontier {
                expected: frontier,
                actual: self.entries,
            });
        }
        if values.shape().len() != 2
            || values.dim(0) <= 0
            || values.dim(1) != self.spec.width
            || values.element_type() != Some(self.spec.element)
        {
            return Err(AppendStreamError::Geometry);
        }
        let added = values.dim(0) as usize;
        let next = self
            .entries
            .checked_add(added)
            .ok_or(AppendStreamError::Geometry)?;
        if next > self.limits.entries {
            return Err(AppendStreamError::Budget {
                resource: "entries",
                required: next as u64,
                limit: self.limits.entries as u64,
            });
        }
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
        if range.start >= range.end || range.end > self.entries {
            return Err(AppendStreamError::Range);
        }
        if range.len() > self.limits.read_entries {
            return Err(AppendStreamError::Budget {
                resource: "read entries",
                required: range.len() as u64,
                limit: self.limits.read_entries as u64,
            });
        }
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
