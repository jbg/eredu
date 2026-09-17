//! Borrowed host slots and a scalar plan for one original prepared input source.
//! This source boundary does not fund earlier processor buffers or native work.
use eredu_core::{InputExtent, InputMetadataKey, InputModality, InputPayloadKind};

mod plan;
pub(crate) use plan::{Counts, KEYS};
pub use plan::{HostInputPlanError, PreparedHostInputPlan};
mod storage;
pub use storage::PreparedHostPart;
pub(crate) use storage::{CompileFailure, Storage};

/// Immutable logical row-major values retained by the original host compiler.
#[derive(Debug, Clone, Copy)]
pub enum HostTensorValues<'a> {
    /// Token IDs or unsigned values.
    U32(&'a [u32]),
    /// Signed metadata or values.
    I32(&'a [i32]),
    /// Floating-point patches or projected inputs.
    F32(&'a [f32]),
    /// Boolean processor metadata, including exact audio masks.
    Bool(&'a [bool]),
}
impl HostTensorValues<'_> {
    /// Exact number of supplied values, without allocation.
    pub fn len(self) -> usize {
        match self {
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }
    /// Whether this value slice is empty.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}
/// Borrowed immutable numerical slot. Shape and values remain caller owned.
#[derive(Debug, Clone, Copy)]
pub struct HostTensorView<'a> {
    /// Logical row-major shape.
    pub shape: &'a [usize],
    /// Exact typed values.
    pub values: HostTensorValues<'a>,
}
/// Concrete borrowed input for scalar original-source planning. It carries no
/// callback, execution grant, native tensor or independently chosen byte count.
#[derive(Debug, Clone, Copy)]
pub struct HostInputPart<'a> {
    /// Actual input modality.
    pub modality: InputModality,
    /// Semantic role of the payload.
    pub kind: InputPayloadKind,
    /// Actual payload values and shape.
    pub payload: HostTensorView<'a>,
    /// Unique metadata slots; original storage canonicalizes known key order.
    pub metadata: &'a [(InputMetadataKey, HostTensorView<'a>)],
    /// Actual host-known semantic extents.
    pub extents: &'a [InputExtent],
}
/// Borrowing view for shared ordinary lowering. Implementing this trait confers
/// no original source custody; the original compiler accepts only concrete parts.
pub trait HostInputPartView {
    /// Input modality.
    fn modality(&self) -> InputModality;
    /// Payload role.
    fn kind(&self) -> InputPayloadKind;
    /// Payload values and shape.
    fn payload(&self) -> HostTensorView<'_>;
    /// Metadata in the source's order, borrowed without a staging collection.
    fn metadata(&self) -> impl ExactSizeIterator<Item = (InputMetadataKey, HostTensorView<'_>)>;
    /// Host-known extents.
    fn extents(&self) -> &[InputExtent];
}
impl HostInputPartView for HostInputPart<'_> {
    fn modality(&self) -> InputModality {
        self.modality
    }
    fn kind(&self) -> InputPayloadKind {
        self.kind
    }
    fn payload(&self) -> HostTensorView<'_> {
        self.payload
    }
    fn metadata(&self) -> impl ExactSizeIterator<Item = (InputMetadataKey, HostTensorView<'_>)> {
        self.metadata.iter().copied()
    }
    fn extents(&self) -> &[InputExtent] {
        self.extents
    }
}
