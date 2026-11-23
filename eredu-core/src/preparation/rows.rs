//! Exact native row-decoder geometry and support, independent of bank policy.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    store::{StoreError, TensorMetadata},
};
use eredu_nn::{
    mechanism_memory::MechanismMemoryContract, tensor_movement::IndexedMovement,
    DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, ParameterId, Tensor,
    TensorElementType,
};

use crate::{
    checkpoint_mechanisms::CheckpointParameterBackend,
    residency::{
        observations::DenseResidencyMechanisms, AddressableStorageCapabilities, ParameterLoan,
    },
    BackendFailure,
};

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

/// Source representation used by a scalar recipe, without artifact identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowScaleSource {
    /// Logical and physical tensor geometry. Shard paths are discarded at admission.
    pub metadata: TensorMetadata,
    /// Exact container encoding required by the materializer.
    pub encoding: eredu_checkpoint::SourceTensorEncoding,
}

/// Native temporary storage for one exact prepared row contract. These are
/// conservative peaks, excluding residency, retained scalars and portable output
/// planning, which the selection driver accounts separately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowLookupWorkspace {
    /// Decode/copy workspace for the largest permitted acquisition.
    pub decode_bytes: u64,
    /// Workspace for materializing the exact scalar recipe during binding.
    pub scalar_bytes: u64,
}

/// Exact encoding and tensor geometry for a compact row decode.
#[derive(Clone, Copy)]
pub struct RowDecodeGeometry<'a> {
    /// Decoded row width.
    pub dimensions: i32,
    /// Native source encoding.
    pub encoding: &'a RowEncoding,
    /// Requested native output representation.
    pub output_type: TensorElementType,
}

/// Exact bounded work for a single invocation of a row decoder.
pub struct RowDecodeRequest<'a> {
    /// Maximum number of distinct rows in one acquisition.
    pub rows: u64,
    /// Encoded bytes in one row.
    pub encoded_row_bytes: u64,
    /// Exact native decode geometry.
    pub geometry: RowDecodeGeometry<'a>,
    /// Scalar recipe and source headers if an independently bound scalar is needed.
    pub scalar: Option<RowScalarMaterialization<'a>>,
}

/// Caller-supplied, payload-free recipe inputs for the exact row scale.
pub struct RowScalarMaterialization<'a> {
    /// Exact scalar materialization recipe.
    pub recipe: &'a DerivedWeightRecipe,
    /// Physical encodings and headers addressed by the recipe.
    pub sources: &'a BTreeMap<String, RowScaleSource>,
}

impl RecipeCatalog for RowScalarMaterialization<'_> {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.sources
            .get(key)
            .map(|entry| entry.metadata.clone())
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}

/// Physical support for one exact row decode. The caller selects and admits banks.
pub trait NativeRowLookupSupport {
    /// Supported storage tiers and retention mechanisms.
    fn row_storage(&self) -> Option<AddressableStorageCapabilities>;
    /// Native allocations and lifetimes for one exact decode, enriching its logical tensor contract.
    fn row_decode_memory(
        &self,
        _request: RowDecodeRequest<'_>,
        logical_memory: MechanismMemoryContract,
    ) -> Result<MechanismMemoryContract, BackendFailure> {
        Ok(logical_memory)
    }
    /// Exact native workspace or absence of a compatible decoder.
    fn row_workspace(
        &self,
        request: RowDecodeRequest<'_>,
    ) -> Result<Option<RowLookupWorkspace>, BackendFailure>;
}

/// Native operations over an already admitted compact row acquisition.
pub trait RowDecode<B: NeuralBackend, N: DenseResidencyMechanisms> {
    /// Failure at the row mechanism boundary.
    type Error: std::error::Error + Send + Sync + From<N::Error> + From<BackendFailure> + 'static;

    /// Native protection for the acquired rows.
    type Loan: ParameterLoan<B::Tensor, <B::Tensor as Tensor>::Context, Error = Self::Error>;

    /// Opens a scope for the exact acquired rows.
    fn begin_loan(
        transfer: N::Transfer,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Loan, Self::Error>;

    /// Reports the exact decoding workspace for the supplied compact shape.
    fn workspace(
        &self,
        rows: u64,
        row_bytes: u64,
        geometry: RowDecodeGeometry<'_>,
    ) -> Result<u64, Self::Error>;

    /// Decodes an admitted range while the caller retains its loan.
    fn decode(
        &mut self,
        loan: &mut Self::Loan,
        binding: &str,
        geometry: RowDecodeGeometry<'_>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Supplies exact native decode and scalar materialization mechanisms.
pub trait RowDecodeBinding<B, N, D>
where
    B: CheckpointParameterBackend,
    N: DenseResidencyMechanisms,
    D: RowDecode<B, N>,
{
    /// Side-effect-free decoder and workspace capabilities.
    fn support(&self) -> impl NativeRowLookupSupport;

    /// Native read and conversion scope for the selected scalar.
    type ParameterReads: crate::checkpoint_mechanisms::ParameterReadScope<Value = B::Tensor>;

    /// Opens the read scope on the exact source and destination contexts.
    fn parameter_reads(&self) -> Self::ParameterReads;

    /// Retains the native scalar, when selected, in an otherwise unbound decoder.
    fn decoder(
        &self,
        scale: Option<(eredu_checkpoint::recipe::RecipeDtype, B::Tensor)>,
    ) -> Result<D, D::Error>;
}

/// Native mechanisms used by the portable bank and row owners.
pub trait NativeRoutedStorage<B>
where
    B: GroupedNeuralBackend
        + DistributedNeuralBackend
        + crate::checkpoint_mechanisms::CheckpointParameterBackend,
{
    /// Native grouped tensor movement.
    type Movement: IndexedMovement<B, Error: std::fmt::Display> + 'static;
    /// Native decode for an admitted row range.
    type Decoder: RowDecode<B, Self::BankMechanisms, Error = Self::StorageError>;
    /// Native context and decoder setup for admitted rows.
    type RowBinding: RowDecodeBinding<B, Self::BankMechanisms, Self::Decoder>;
    /// Native storage binding failure.
    type StorageError: std::fmt::Display + From<crate::parameters::ParameterError>;

    /// Supplies exact native row lookup capabilities.
    fn row_lookup_support(&self) -> impl crate::preparation::rows::NativeRowLookupSupport;

    /// Native checkpoint parameter format support for this pool.
    type BankParameters: crate::checkpoint_mechanisms::CheckpointParameterBackend;

    /// Native storage and transfer mechanisms for this bank.
    type BankMechanisms: DenseResidencyMechanisms
        + crate::residency::NativeBankMechanisms<
            Backend = B,
            Storage: crate::residency::ResidencyStorage<Backend = Self::BankParameters>,
        >;

    /// Exact native placement for the admitted bank.
    fn bank_residency_context(
        &self,
    ) -> <Self::BankMechanisms as crate::residency::ResidencyStorageMechanisms>::Context;

    /// Supplies native recipe lowering and bounded conversion.
    fn bank_materializer(
        &self,
    ) -> impl crate::checkpoint_mechanisms::CheckpointEncodingMaterializer<
        Parameters = Self::BankParameters,
    >;

    /// Supplies the queue-bound indexed movement primitives.
    fn indexed_movement(&self) -> Self::Movement;

    /// Supplies tensor conversion and scalar materialization on native queues.
    fn row_binding(&self) -> Self::RowBinding;
}
