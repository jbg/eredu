//! Bounded, exact-integer row lookup over generic parameter-bank storage.
use crate::{ParameterBank, ParameterBankAccess, ParameterBankAcquisition, ParameterBankKey};
use eredu_nn::{NeuralBackend, ParameterId, Tensor, TensorElementType};

mod descriptor;
pub use descriptor::{
    RowLookupDescriptor, RowLookupDescriptors, RowScaleDescriptor, RowScaleSource,
};

mod collection;
pub use collection::{PreparedRowLookups, RowLookupRequirements};

mod memory;
mod parallel;
pub use parallel::TensorParallelRowLookup;
mod selection;
pub use selection::{
    RowLookupMechanismSupport, RowLookupSelectionError, RowLookupWorkspace, SelectedRowLookupPlans,
    SelectedRowLookupRequirements, SelectedRowLookups,
};

mod prepared;
pub use prepared::{PreparedRowLookup, PreparedRowScale};

/// Architecture-declared row encoding; companions have authoritative identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowEncoding {
    /// Ordinary floating-point matrix rows.
    Dense,
    /// E4M3 bytes multiplied in FP32 by one shared floating-point scalar.
    ScalarE4M3 {
        /// Exact shared scale parameter, retained separately from table rows.
        scale: ParameterId,
    },
    /// Independently decodable GGML row blocks.
    Gguf {
        /// Canonical block encoding.
        encoding: eredu_gguf::GgmlType,
        /// Byte order of the retained payload.
        endian: eredu_gguf::Endian,
    },
}

/// Exact logical geometry and generic bank ownership for one row source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowLookupSpec {
    /// Authoritative table identity.
    pub parameter: ParameterId,
    /// Independently addressable bank ordinal.
    pub bank: usize,
    /// Owning execution-unit ordinal.
    pub unit: usize,
    /// Global row count, independent of native index tensor widths.
    pub rows: u64,
    /// Decoded row width.
    pub dimensions: i32,
    /// Encoding and exact companion identities.
    pub encoding: RowEncoding,
    /// Arithmetic boundary after decoding/scaling.
    pub output_type: TensorElementType,
}
impl RowLookupSpec {
    /// Rejects empty geometry, unsupported output arithmetic, and address overflow.
    pub fn validate(&self) -> Result<(), RowLookupError> {
        if self.rows == 0
            || self.rows > usize::MAX as u64
            || self.dimensions <= 0
            || !matches!(
                self.output_type,
                TensorElementType::F32 | TensorElementType::F16 | TensorElementType::Bf16
            )
        {
            return Err(RowLookupError::Geometry);
        }
        if let RowEncoding::Gguf { encoding, .. } = self.encoding {
            let Ok((block, _)) = encoding.block_and_bytes() else {
                return Err(RowLookupError::Geometry);
            };
            if self.dimensions as u64 % block != 0 {
                return Err(RowLookupError::Geometry);
            }
        }
        Ok(())
    }
    fn key(&self, row: u64) -> ParameterBankKey {
        ParameterBankKey::new(self.bank, self.unit, row as usize)
    }
    fn output_bytes(&self) -> u64 {
        self.dimensions as u64
            * if self.output_type == TensorElementType::F32 {
                4
            } else {
                2
            }
    }
}

/// Independent limits for request planning, source residency and transient output.
/// Native transfer/conversion reservations remain enforced by the shared bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLookupLimits {
    /// Includes duplicates; bounds host sort/count/permutation metadata.
    pub requests: usize,
    /// Maximum simultaneously acquired distinct rows.
    pub rows_per_acquisition: usize,
    /// Maximum encoded payload acquired in one batch.
    pub acquisition_bytes: u64,
    /// Bound for host planning arrays (excluding caller-owned token IDs).
    pub host_bytes: u64,
    /// Bound for completed chunks, concatenation and order-restored output.
    pub output_bytes: u64,
}
impl RowLookupLimits {
    /// Checks nonempty addressable request geometry and buffer allowances.
    pub fn validate(&self) -> Result<(), RowLookupError> {
        if self.requests == 0
            || self.requests > i32::MAX as usize
            || self.rows_per_acquisition == 0
            || self.acquisition_bytes == 0
            || self.host_bytes == 0
            || self.output_bytes == 0
        {
            return Err(RowLookupError::Geometry);
        }
        Ok(())
    }
}

/// Portable failures preserve native causes without exposing native error types.
#[derive(Debug, thiserror::Error)]
pub enum RowLookupError {
    /// Geometry or arithmetic declaration cannot be represented.
    #[error("invalid row lookup geometry or encoding")]
    Geometry,
    /// The prepared provider does not own the requested parameter.
    #[error("no row lookup provider for {0}")]
    Missing(ParameterId),
    /// A request differs from the exact prepared specification.
    #[error("row lookup specification differs from prepared parameter {0}")]
    Specification(ParameterId),
    /// Original integer ID is outside the table.
    #[error("row {row} is outside 0..{rows}")]
    OutOfRange {
        /// Original integer ID.
        row: u64,
        /// Admitted row count.
        rows: u64,
    },
    /// A request exceeds a specific admitted resource.
    #[error("row lookup {resource} requires {required} but admits {limit}")]
    Budget {
        /// Resource being reserved.
        resource: &'static str,
        /// Required quantity.
        required: u64,
        /// Admitted quantity.
        limit: u64,
    },
    /// Another participant rejected the shared row lookup before data exchange.
    #[error("another tensor-parallel participant failed its row lookup")]
    ParallelPeerFailure,
    /// Native storage, decoding or completion failure.
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
    /// Neutral tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] eredu_nn::Error),
}
impl RowLookupError {
    fn bank(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Backend(eredu_core::BackendFailure::from_error(error).with_operation("row lookup"))
    }
}

/// Row decoding shares acquisition and accounting with grouped parameter banks.
pub trait RowLookupBank<B: NeuralBackend>: ParameterBank<B> {
    /// Decodes acquired rows in acquisition order and completes their use before
    /// returning. On failure unresolved submissions must retain every lease.
    /// The output owns its compact storage independently of resident source rows.
    fn rows(
        &mut self,
        acquisition: Self::Acquisition,
        spec: &RowLookupSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Shared execution-provider surface; original integer IDs never pass through floats.
pub trait RowLookupProvider<B: NeuralBackend> {
    /// Whether this provider owns the exact declared parameter.
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool;

    /// Looks up exact host IDs, restoring their order and multiplicity.
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RowLookupError>;
}

/// Portable request planning and bounded acquisition over one retained row bank.
pub struct BoundedRowLookup<K> {
    bank: K,
    spec: RowLookupSpec,
    limits: RowLookupLimits,
}
impl<K> BoundedRowLookup<K> {
    /// Retains the exact table declaration and explicit limits.
    pub fn new(
        bank: K,
        spec: RowLookupSpec,
        limits: RowLookupLimits,
    ) -> Result<Self, RowLookupError> {
        spec.validate()?;
        limits.validate()?;
        Ok(Self { bank, spec, limits })
    }
    /// Reads shared residency telemetry through the underlying generic bank.
    pub fn bank(&self) -> &K {
        &self.bank
    }
    /// Mutably accesses the underlying bank for native lifecycle operations.
    pub fn bank_mut(&mut self) -> &mut K {
        &mut self.bank
    }
}
fn admit(resource: &'static str, required: u64, limit: u64) -> Result<(), RowLookupError> {
    if required > limit {
        Err(RowLookupError::Budget {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
impl<B: NeuralBackend, K: RowLookupBank<B>> RowLookupProvider<B> for BoundedRowLookup<K> {
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool {
        parameter == &self.spec.parameter
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RowLookupError> {
        if spec != &self.spec {
            return Err(RowLookupError::Specification(spec.parameter.clone()));
        }
        admit("requests", rows.len() as u64, self.limits.requests as u64)?;
        // Empty requests have no meaningful injection; callers skip their unit.
        if rows.is_empty() {
            return Err(RowLookupError::Geometry);
        }
        // Unique IDs, counts, keys, permutation, native integer indices and chunk
        // descriptors coexist. This bound uses per-request capacity, not table size.
        let host = (rows.len() as u64)
            .checked_mul(128)
            .ok_or(RowLookupError::Geometry)?;
        admit("host planning bytes", host, self.limits.host_bytes)?;
        let output = (rows.len() as u64)
            .checked_mul(spec.output_bytes())
            .and_then(|x| x.checked_mul(3))
            .ok_or(RowLookupError::Geometry)?;
        admit("output bytes", output, self.limits.output_bytes)?;
        for &row in rows {
            if row >= spec.rows {
                return Err(RowLookupError::OutOfRange {
                    row,
                    rows: spec.rows,
                });
            }
        }
        let mut unique = rows.to_vec();
        unique.sort_unstable();
        unique.dedup();
        let mut demands = vec![0u64; unique.len()];
        let restore = rows
            .iter()
            .map(|row| {
                let i = unique.binary_search(row).expect("validated requested row");
                demands[i] += 1;
                i as i32
            })
            .collect::<Vec<_>>();
        // Validate every member and batch before any acquisition, so malformed IDs
        // and budget exhaustion cannot leave a partially executed lookup.
        let mut batches = Vec::new();
        let mut start = 0;
        let mut bytes = 0u64;
        for (i, &row) in unique.iter().enumerate() {
            let size = self
                .bank
                .member_bytes(spec.key(row))
                .ok_or_else(|| RowLookupError::Missing(spec.parameter.clone()))?;
            if size == 0 {
                return Err(RowLookupError::Geometry);
            }
            admit("acquisition bytes", size, self.limits.acquisition_bytes)?;
            if i > start
                && (i - start == self.limits.rows_per_acquisition
                    || bytes > self.limits.acquisition_bytes - size)
            {
                batches.push(start..i);
                start = i;
                bytes = 0;
            }
            bytes += size;
        }
        batches.push(start..unique.len());
        let mut outputs = Vec::with_capacity(batches.len());
        for batch in batches {
            let entries = batch
                .clone()
                .map(|i| (spec.key(unique[i]), demands[i]))
                .collect::<Vec<_>>();
            let acquisition = self
                .bank
                .acquire(ParameterBankAcquisition::new(&entries, access), context)
                .map_err(RowLookupError::bank)?;
            let output = self
                .bank
                .rows(acquisition, spec, context)
                .map_err(RowLookupError::bank)?;
            if output.shape() != [batch.len() as i32, spec.dimensions] {
                return Err(RowLookupError::Geometry);
            }
            outputs.push(output);
        }
        let compact = B::Tensor::concatenate(&outputs, 0, context)?;
        let indices = B::Tensor::from_i32_slice(&restore, &[restore.len() as i32], context)?;
        Ok(compact.take_axis(&indices, 0, context)?)
    }
}

/// Compact set of table providers, one entry per parameter, never per row.
pub struct RowLookupProviders<P> {
    providers: std::collections::BTreeMap<ParameterId, P>,
}
impl<P> RowLookupProviders<P> {
    /// Retains unique authoritative table identities.
    pub fn new(
        providers: impl IntoIterator<Item = (ParameterId, P)>,
    ) -> Result<Self, RowLookupError> {
        let mut result = std::collections::BTreeMap::new();
        for (id, provider) in providers {
            if result.insert(id.clone(), provider).is_some() {
                return Err(RowLookupError::Specification(id));
            }
        }
        Ok(Self { providers: result })
    }
    /// Iterates selected row mechanisms for accounting and lifecycle inspection.
    pub fn providers(&self) -> &std::collections::BTreeMap<ParameterId, P> {
        &self.providers
    }
}
impl<B: NeuralBackend, P: RowLookupProvider<B>> RowLookupProvider<B> for RowLookupProviders<P> {
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool {
        self.providers
            .get(parameter)
            .is_some_and(|provider| provider.has_row_parameter(parameter))
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RowLookupError> {
        self.providers
            .get_mut(&spec.parameter)
            .ok_or_else(|| RowLookupError::Missing(spec.parameter.clone()))?
            .lookup_rows(spec, rows, access, context)
    }
}

impl<B: NeuralBackend, P: RowLookupProvider<B>> RowLookupProvider<B> for Option<P> {
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool {
        self.as_ref()
            .is_some_and(|provider| provider.has_row_parameter(parameter))
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RowLookupError> {
        self.as_mut()
            .ok_or_else(|| RowLookupError::Missing(spec.parameter.clone()))?
            .lookup_rows(spec, rows, access, context)
    }
}

/// Explicit empty mechanism for executions with no declared row parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRowLookups;
impl<B: NeuralBackend> RowLookupProvider<B> for NoRowLookups {
    fn has_row_parameter(&self, _: &ParameterId) -> bool {
        false
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        _: &[u64],
        _: ParameterBankAccess,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RowLookupError> {
        Err(RowLookupError::Missing(spec.parameter.clone()))
    }
}
