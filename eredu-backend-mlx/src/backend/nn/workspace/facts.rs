//! Shared count, caller-destination and ordinary storage policies. All format
//! arguments originate in the closed native fact helpers; no caller formatter,
//! tensor, stream or callback is accepted by this module's public boundary.
use super::*;
use std::fmt::{self, Write};

/// Concrete fixed cause in the selected native cold fact companion.
/// These values own no native error, formatted diagnostic, or dynamic storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MlxWorkspaceFactCause {
    /// A checked native descriptor or arithmetic rule failed.
    #[error("{0}")]
    Descriptor(&'static str),
    /// The common represented-layout kernel failed.
    #[error(transparent)]
    Layout(#[from] WorkspaceLayoutError),
    /// Native geometry cannot be represented by the actual integer ABI.
    #[error(transparent)]
    Integer(#[from] std::num::TryFromIntError),
    /// The borrowed physical linear description failed its shared validator.
    #[error(transparent)]
    LinearFormat(#[from] eredu_nn::LinearFormatValidationError),
    /// The borrowed embedding policy failed its shared validator.
    #[error(transparent)]
    Embedding(#[from] eredu_nn::EmbeddingValidationError),
    /// Closed host-generated initialization geometry failed.
    #[error(transparent)]
    F32Initialization(#[from] eredu_nn::F32InitializationError),
    /// The borrowed gated-product policy failed its shared validator.
    #[error(transparent)]
    GatedProduct(#[from] eredu_nn::GatedProductValidationError),
    /// Compact projection reconstruction failed its fixed source geometry.
    #[error(transparent)]
    ProjectionObservation(#[from] eredu_nn::ProjectionObservationError),
    /// The shared causal or sliding attention position geometry failed.
    #[error(transparent)]
    AttentionGeometry(#[from] eredu_nn::operation_geometry::AttentionGeometryError),
    /// Normalization construction rejected its exact retained scalar policy.
    #[error(transparent)]
    Normalization(#[from] eredu_nn::NormalizationValidationError),
    /// The retained multi-stream construction has invalid scalar geometry.
    #[error(transparent)]
    HyperConnection(#[from] eredu_nn::HyperConnectionValidationError),
    /// The retained final-stream construction has invalid scalar geometry.
    #[error(transparent)]
    HyperHead(#[from] eredu_nn::HyperHeadValidationError),
    /// The borrowed masked-output geometry failed its shared shape rules.
    #[error(transparent)]
    MaskedOutput(#[from] eredu_nn::operation_geometry::MaskedOutputGeometryError),
    /// The normalized rotary algorithm rejected its scalar policy.
    #[error(transparent)]
    RotaryAlgorithm(#[from] eredu_nn::RotaryValidationError),
    /// The original borrowed rotary source policy failed its fixed kernel.
    #[error(transparent)]
    RotaryTable(#[from] eredu_nn::multimodal::RotaryTableError),
    /// A caller destination has the wrong exact length.
    #[error(transparent)]
    Destination(#[from] WorkspaceFactDestinationError),
    /// Grouped linear source geometry or projection identity is invalid.
    #[error(transparent)]
    GroupedLinear(#[from] eredu_nn::GroupedLinearValidationError),
    /// Grouped gated-product source geometry or parameter identity is invalid.
    #[error(transparent)]
    GroupedBank(#[from] eredu_nn::GroupedBankValidationError),
    /// Grouped Relu2 source geometry or parameter identity is invalid.
    #[error(transparent)]
    GroupedRelu2(#[from] eredu_nn::GroupedRelu2ValidationError),
    /// Physical row partitions or scale-row arithmetic failed.
    #[error(transparent)]
    LinearRows(#[from] eredu_nn::LinearRowError),
    /// Selector construction rejected its exact borrowed policy/source.
    #[error(transparent)]
    Selector(#[from] eredu_nn::SelectorValidationError),
    /// The shared intervention driver rejected its control/effective decision.
    #[error(transparent)]
    RoutingInvalid(#[from] eredu_nn::routing_intervention::RoutingInvalidCause),
    /// The number of metadata cells or UTF-8 bytes cannot fit in `usize`.
    #[error("native workspace fact population overflow")]
    PopulationOverflow,
}

impl MlxWorkspaceFactCause {
    pub(super) fn ordinary(self) -> Error {
        match self {
            Self::Layout(error) => error.into(),
            Self::Integer(error) => Error::backend_retained_source(error),
            Self::F32Initialization(error) => Error::backend_retained_source(error),
            Self::ProjectionObservation(error) => Error::backend_retained_source(error),
            Self::RotaryTable(error) => Error::backend_retained_source(error),
            other => Error::backend(other),
        }
    }
}

/// Fixed native fact failure and its actual shared-driver context.
/// The routing prefix is added only by the closed selector's single driver
/// invocation; its primitive Counter methods never invoke another router.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxWorkspaceFactError {
    cause: MlxWorkspaceFactCause,
    routing_native: bool,
}
impl std::fmt::Display for MlxWorkspaceFactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.routing_native {
            f.write_str("native routing operation failed: ")?;
        }
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for MlxWorkspaceFactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl MlxWorkspaceFactError {
    /// Borrows the original fixed cause without formatting or allocating it.
    pub const fn cause(&self) -> &MlxWorkspaceFactCause {
        &self.cause
    }
    pub(super) const POPULATION_OVERFLOW: Self = Self {
        cause: MlxWorkspaceFactCause::PopulationOverflow,
        routing_native: false,
    };
    pub(super) fn ordinary(self) -> Error {
        if self.routing_native {
            // The old driver adapter used Error::backend on its wrapper,
            // retaining the exact formatted message with no source object.
            Error::backend(self)
        } else {
            self.cause.ordinary()
        }
    }
    pub(super) fn routing_invalid(
        cause: eredu_nn::routing_intervention::RoutingInvalidCause,
    ) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::RoutingInvalid(cause),
            routing_native: false,
        }
    }
    pub(super) fn in_routing(self) -> Self {
        // Only selector::Counter primitive failures reach this site. None of
        // those methods call routing execution, so the wrapper occurs once.
        Self {
            routing_native: true,
            ..self
        }
    }
    pub(super) fn descriptor(cause: &'static str) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::Descriptor(cause),
            routing_native: false,
        }
    }
    pub(super) fn layout(cause: WorkspaceLayoutError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::Layout(cause),
            routing_native: false,
        }
    }
    pub(super) fn integer(cause: std::num::TryFromIntError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::Integer(cause),
            routing_native: false,
        }
    }
    pub(super) fn linear_format(cause: eredu_nn::LinearFormatValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::LinearFormat(cause),
            routing_native: false,
        }
    }
    pub(super) fn embedding(cause: eredu_nn::EmbeddingValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::Embedding(cause),
            routing_native: false,
        }
    }
    pub(super) fn f32_initialization(cause: eredu_nn::F32InitializationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::F32Initialization(cause),
            routing_native: false,
        }
    }
    pub(super) fn gated_product(cause: eredu_nn::GatedProductValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::GatedProduct(cause),
            routing_native: false,
        }
    }
    pub(super) fn projection_observation(cause: eredu_nn::ProjectionObservationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::ProjectionObservation(cause),
            routing_native: false,
        }
    }
    pub(super) fn attention_geometry(
        cause: eredu_nn::operation_geometry::AttentionGeometryError,
    ) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::AttentionGeometry(cause),
            routing_native: false,
        }
    }
    pub(super) fn normalization(cause: eredu_nn::NormalizationValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::Normalization(cause),
            routing_native: false,
        }
    }
    pub(super) fn hyper_connection(cause: eredu_nn::HyperConnectionValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::HyperConnection(cause),
            routing_native: false,
        }
    }
    pub(super) fn hyper_head(cause: eredu_nn::HyperHeadValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::HyperHead(cause),
            routing_native: false,
        }
    }
    pub(super) fn masked_output(
        cause: eredu_nn::operation_geometry::MaskedOutputGeometryError,
    ) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::MaskedOutput(cause),
            routing_native: false,
        }
    }
    pub(super) fn rotary_table(cause: eredu_nn::multimodal::RotaryTableError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::RotaryTable(cause),
            routing_native: false,
        }
    }
    pub(super) fn destination(cause: WorkspaceFactDestinationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::Destination(cause),
            routing_native: false,
        }
    }
}
impl From<WorkspaceLayoutError> for MlxWorkspaceFactError {
    fn from(cause: WorkspaceLayoutError) -> Self {
        Self::layout(cause)
    }
}
impl From<std::num::TryFromIntError> for MlxWorkspaceFactError {
    fn from(cause: std::num::TryFromIntError) -> Self {
        Self::integer(cause)
    }
}
impl From<eredu_nn::LinearFormatValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::LinearFormatValidationError) -> Self {
        Self::linear_format(cause)
    }
}
impl From<eredu_nn::EmbeddingValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::EmbeddingValidationError) -> Self {
        Self::embedding(cause)
    }
}
impl From<eredu_nn::F32InitializationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::F32InitializationError) -> Self {
        Self::f32_initialization(cause)
    }
}
impl From<eredu_nn::GatedProductValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::GatedProductValidationError) -> Self {
        Self::gated_product(cause)
    }
}
impl From<eredu_nn::ProjectionObservationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::ProjectionObservationError) -> Self {
        Self::projection_observation(cause)
    }
}
impl From<eredu_nn::operation_geometry::AttentionGeometryError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::operation_geometry::AttentionGeometryError) -> Self {
        Self::attention_geometry(cause)
    }
}
impl From<eredu_nn::NormalizationValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::NormalizationValidationError) -> Self {
        Self::normalization(cause)
    }
}
impl From<eredu_nn::HyperConnectionValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::HyperConnectionValidationError) -> Self {
        Self::hyper_connection(cause)
    }
}
impl From<eredu_nn::HyperHeadValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::HyperHeadValidationError) -> Self {
        Self::hyper_head(cause)
    }
}
impl From<eredu_nn::operation_geometry::MaskedOutputGeometryError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::operation_geometry::MaskedOutputGeometryError) -> Self {
        Self::masked_output(cause)
    }
}
impl From<eredu_nn::multimodal::RotaryTableError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::multimodal::RotaryTableError) -> Self {
        Self::rotary_table(cause)
    }
}
impl From<WorkspaceFactDestinationError> for MlxWorkspaceFactError {
    fn from(cause: WorkspaceFactDestinationError) -> Self {
        Self::destination(cause)
    }
}

impl From<eredu_nn::SelectorValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::SelectorValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::Selector(cause),
            routing_native: false,
        }
    }
}

impl From<eredu_nn::GroupedLinearValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::GroupedLinearValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::GroupedLinear(cause),
            routing_native: false,
        }
    }
}
impl From<eredu_nn::GroupedBankValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::GroupedBankValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::GroupedBank(cause),
            routing_native: false,
        }
    }
}
impl From<eredu_nn::GroupedRelu2ValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::GroupedRelu2ValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::GroupedRelu2(cause),
            routing_native: false,
        }
    }
}
impl From<eredu_nn::LinearRowError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::LinearRowError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::LinearRows(cause),
            routing_native: false,
        }
    }
}

impl From<eredu_nn::RotaryValidationError> for MlxWorkspaceFactError {
    fn from(cause: eredu_nn::RotaryValidationError) -> Self {
        Self {
            cause: MlxWorkspaceFactCause::RotaryAlgorithm(cause),
            routing_native: false,
        }
    }
}

pub(super) type FactResult<T> = Result<T, MlxWorkspaceFactError>;

/// Either borrowed explicit candidates or all ordinals in a checked range.
/// Range storage is independent of rank/input count and needs no scratch list.
#[derive(Clone, Copy)]
pub(super) enum Aliases<'a> {
    Slice(&'a [usize]),
    Range { start: usize, end: usize },
}

impl Aliases<'_> {
    fn len(self) -> FactResult<usize> {
        match self {
            Self::Slice(values) => Ok(values.len()),
            Self::Range { start, end } => end
                .checked_sub(start)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW),
        }
    }
    fn at(self, ordinal: usize) -> usize {
        match self {
            Self::Slice(values) => values[ordinal],
            Self::Range { start, .. } => start + ordinal,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Output<'a> {
    Allocate(u64),
    AliasInput(usize),
    AliasOutput(usize),
    AllocateOrAliasInputs { bytes: u64, inputs: Aliases<'a> },
}

enum Storage<'a> {
    Count,
    Fixed(WorkspaceEffectDestination<'a>),
    // Only the ordinary adapter selects this policy. It preserves the actual
    // owning Vec/nested Vec/String result, outside the fixed companion.
    Ordinary {
        outputs: Vec<WorkspaceOutputStorage>,
        assumptions: String,
    },
}

pub(super) struct Emitter<'a> {
    storage: Storage<'a>,
    outputs: usize,
    aliases: usize,
    first_output: Option<WorkspaceOutputEffect>,
    allocated_output_bytes: Option<u64>,
}

impl<'a> Emitter<'a> {
    pub(super) fn count() -> Self {
        Self {
            storage: Storage::Count,
            outputs: 0,
            aliases: 0,
            first_output: None,
            allocated_output_bytes: Some(0),
        }
    }
    pub(super) fn fixed(destination: WorkspaceEffectDestination<'a>) -> Self {
        Self {
            storage: Storage::Fixed(destination),
            outputs: 0,
            aliases: 0,
            first_output: None,
            allocated_output_bytes: Some(0),
        }
    }
    fn ordinary() -> Self {
        Self {
            storage: Storage::Ordinary {
                outputs: Vec::new(),
                assumptions: String::new(),
            },
            outputs: 0,
            aliases: 0,
            first_output: None,
            allocated_output_bytes: Some(0),
        }
    }

    /// The actual first effect is retained by the count pass for closed
    /// single-output child equations; no output vector or replay is needed.
    pub(super) fn first_output(&self) -> Option<WorkspaceOutputEffect> {
        self.first_output
    }

    /// Sum of actual allocation effects, counting every independent output and
    /// no aliases. Overflow is retained without changing the ordinary emitter;
    /// a resident population consumer must reject it before admission.
    pub(super) fn allocated_output_bytes(&self) -> Option<u64> {
        self.allocated_output_bytes
    }

    pub(super) fn output(&mut self, output: Output<'_>) -> FactResult<()> {
        let next = self
            .outputs
            .checked_add(1)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        let candidates = match output {
            Output::AllocateOrAliasInputs { inputs, .. } => inputs,
            _ => Aliases::Slice(&[]),
        };
        let count = candidates.len()?;
        let end = self
            .aliases
            .checked_add(count)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        let effect = match output {
            Output::Allocate(bytes) => WorkspaceOutputEffect::Allocate(bytes),
            Output::AliasInput(input) => WorkspaceOutputEffect::AliasInput(input),
            Output::AliasOutput(previous) => WorkspaceOutputEffect::AliasOutput(previous),
            Output::AllocateOrAliasInputs { bytes, .. } => {
                WorkspaceOutputEffect::AllocateOrAliasInputs {
                    bytes,
                    alias_start: self.aliases,
                    alias_count: count,
                }
            }
        };
        if self.outputs == 0 {
            self.first_output = Some(effect);
        }
        match &mut self.storage {
            Storage::Count => {}
            Storage::Fixed(destination) => {
                // The immutable count pass validated these exact bounds before
                // constructing this emitter. There is no reserve or fallback.
                destination.outputs[self.outputs] = effect;
                for ordinal in 0..count {
                    destination.aliases[self.aliases + ordinal] = candidates.at(ordinal);
                }
            }
            Storage::Ordinary { outputs, .. } => outputs.push(match output {
                Output::Allocate(bytes) => WorkspaceOutputStorage::Allocate(bytes),
                Output::AliasInput(input) => WorkspaceOutputStorage::AliasInput(input),
                Output::AliasOutput(previous) => WorkspaceOutputStorage::AliasOutput(previous),
                Output::AllocateOrAliasInputs { bytes, .. } => {
                    WorkspaceOutputStorage::AllocateOrAliasInputs {
                        bytes,
                        inputs: (0..count).map(|ordinal| candidates.at(ordinal)).collect(),
                    }
                }
            }),
        }
        let bytes = match output {
            Output::Allocate(bytes) | Output::AllocateOrAliasInputs { bytes, .. } => bytes,
            Output::AliasInput(_) | Output::AliasOutput(_) => 0,
        };
        self.allocated_output_bytes = self
            .allocated_output_bytes
            .and_then(|n| n.checked_add(bytes));
        self.outputs = next;
        self.aliases = end;
        Ok(())
    }

    pub(super) fn finish(
        &mut self,
        scratch_bytes: u64,
        arguments: fmt::Arguments<'_>,
    ) -> FactResult<WorkspaceOperationFacts> {
        let assumption_bytes = match &mut self.storage {
            Storage::Count => text_length(arguments)?,
            Storage::Fixed(destination) => write_text(arguments, destination.assumptions)?,
            Storage::Ordinary { assumptions, .. } => {
                // String's fmt::Write accepts every string. Allocation remains
                // the existing ordinary caller's responsibility.
                assumptions
                    .write_fmt(arguments)
                    .map_err(|_| MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
                assumptions.len()
            }
        };
        Ok(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: self.outputs,
                aliases: self.aliases,
                assumption_bytes,
            },
            scratch_bytes,
        })
    }
}

pub(super) fn ordinary(
    worker: impl FnOnce(&mut Emitter<'_>) -> FactResult<Option<WorkspaceOperationFacts>>,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    ordinary_with(worker, MlxWorkspaceFactError::ordinary)
}

/// Only ordinary adapters use the source-aware diagnostic bridge. The fixed
/// count/fill path never invokes this mapper or reconstructs owned source data.
pub(super) fn ordinary_with(
    worker: impl FnOnce(&mut Emitter<'_>) -> FactResult<Option<WorkspaceOperationFacts>>,
    error: impl FnOnce(MlxWorkspaceFactError) -> Error,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    let mut emitter = Emitter::ordinary();
    let Some(facts) = worker(&mut emitter).map_err(error)? else {
        return Ok(None);
    };
    let Storage::Ordinary {
        outputs,
        assumptions,
    } = emitter.storage
    else {
        unreachable!()
    };
    Ok(Some(WorkspaceOperationBound {
        outputs,
        scratch_bytes: facts.scratch_bytes,
        assumptions,
    }))
}

/// The worker is closed over immutable selected facts and borrowed source
/// views. Its first pass checks every geometry/arithmetic/text term. No caller
/// destination is exposed until the complete count and every exact length pass.
pub(super) fn write(
    worker: impl Fn(&mut Emitter<'_>) -> FactResult<Option<WorkspaceOperationFacts>>,
    destination: WorkspaceEffectDestination<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(expected) = worker(&mut Emitter::count())? else {
        return Ok(None);
    };
    destination.validate(expected.layout)?;
    worker(&mut Emitter::fixed(destination))
}

enum HostStorage<'a> {
    Count,
    Fixed(WorkspaceHostDestination<'a>),
    Ordinary(String),
}

pub(super) struct HostEmitter<'a>(HostStorage<'a>);
impl HostEmitter<'_> {
    pub(super) fn count() -> Self {
        Self(HostStorage::Count)
    }
    pub(super) fn finish(
        &mut self,
        bytes: u64,
        arguments: fmt::Arguments<'_>,
    ) -> FactResult<WorkspaceHostFacts> {
        let assumption_bytes = match &mut self.0 {
            HostStorage::Count => text_length(arguments)?,
            HostStorage::Fixed(destination) => write_text(arguments, destination.assumptions)?,
            HostStorage::Ordinary(assumptions) => {
                assumptions
                    .write_fmt(arguments)
                    .map_err(|_| MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
                assumptions.len()
            }
        };
        Ok(WorkspaceHostFacts {
            bytes,
            assumption_bytes,
        })
    }
}

pub(super) fn ordinary_host(
    worker: impl FnOnce(&mut HostEmitter<'_>) -> FactResult<Option<WorkspaceHostFacts>>,
) -> Result<Option<WorkspaceHostBound>, Error> {
    ordinary_host_with(worker, MlxWorkspaceFactError::ordinary)
}

pub(super) fn ordinary_host_with(
    worker: impl FnOnce(&mut HostEmitter<'_>) -> FactResult<Option<WorkspaceHostFacts>>,
    error: impl FnOnce(MlxWorkspaceFactError) -> Error,
) -> Result<Option<WorkspaceHostBound>, Error> {
    let mut emitter = HostEmitter(HostStorage::Ordinary(String::new()));
    let Some(facts) = worker(&mut emitter).map_err(error)? else {
        return Ok(None);
    };
    let HostStorage::Ordinary(assumptions) = emitter.0 else {
        unreachable!()
    };
    Ok(Some(WorkspaceHostBound {
        bytes: facts.bytes,
        assumptions,
    }))
}

pub(super) fn write_host(
    worker: impl Fn(&mut HostEmitter<'_>) -> FactResult<Option<WorkspaceHostFacts>>,
    destination: WorkspaceHostDestination<'_>,
) -> FactResult<Option<WorkspaceHostFacts>> {
    let Some(expected) = worker(&mut HostEmitter::count())? else {
        return Ok(None);
    };
    destination.validate(expected)?;
    worker(&mut HostEmitter(HostStorage::Fixed(destination)))
}

struct TextWriter<'a> {
    destination: Option<&'a mut [u8]>,
    bytes: usize,
}
impl fmt::Write for TextWriter<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let end = self.bytes.checked_add(text.len()).ok_or(fmt::Error)?;
        if let Some(destination) = &mut self.destination {
            let target = destination.get_mut(self.bytes..end).ok_or(fmt::Error)?;
            target.copy_from_slice(text.as_bytes());
        }
        self.bytes = end;
        Ok(())
    }
}

fn text_length(arguments: fmt::Arguments<'_>) -> FactResult<usize> {
    let mut writer = TextWriter {
        destination: None,
        bytes: 0,
    };
    writer
        .write_fmt(arguments)
        .map_err(|_| MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(writer.bytes)
}
fn write_text(arguments: fmt::Arguments<'_>, destination: &mut [u8]) -> FactResult<usize> {
    let mut writer = TextWriter {
        destination: Some(destination),
        bytes: 0,
    };
    writer
        .write_fmt(arguments)
        .map_err(|_| MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(writer.bytes)
}

pub(super) fn add(a: u64, b: u64) -> FactResult<u64> {
    a.checked_add(b).ok_or(MlxWorkspaceFactError::descriptor(
        "native workspace byte sum overflow",
    ))
}
pub(super) fn mul(a: u64, b: u64) -> FactResult<u64> {
    a.checked_mul(b).ok_or(MlxWorkspaceFactError::descriptor(
        "native workspace byte product overflow",
    ))
}

pub(super) fn buffer_capacity(allocation: NativeAllocationFacts, bytes: u64) -> FactResult<u64> {
    if bytes == 0 && !allocation.cpu_header {
        return Ok(0);
    }
    let page = allocation.page_size();
    let bytes = if allocation.cpu_header {
        add(bytes, std::mem::size_of::<usize>() as u64)?
    } else {
        bytes
    };
    let rounded = if allocation.cpu_header || bytes > page {
        mul(bytes.div_ceil(page), page)?
    } else {
        bytes
    };
    Ok(mul(rounded, 2)?.min(add(rounded, mul(page, 2)?)?) - 1)
}

#[cfg(test)]
mod tests;
