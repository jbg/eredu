//! Metadata execution of the ordinary tensor and neural contracts.
//!
//! This module allocates no tensor values and selects no native device. A
//! concrete mechanism supplies allocation bounds for each exact operation.
//! Unknown bounds remain unknown. The conservative trace retains every unique
//! allocation and every operation's scratch until the completed span boundary;
//! a native scheduler may release them earlier, but that is not assumed here.

use crate::{Error, LinearFormatSpec};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    fmt,
    rc::Rc,
};

mod backend;
mod isolated_copy;
mod layout;
mod representation;
mod shared_shape;
pub use layout::{WorkspaceLayoutError, WorkspaceLayoutView};
pub use representation::{
    WorkspaceFloatingType, WorkspaceParameterRepresentation, WorkspaceRepresentation,
};
mod placement;
mod report;
pub use placement::{
    WorkspaceAllocationPopulation, WorkspacePlacementError, WorkspaceScratchAllocation,
};
mod sampling;
mod shape;
pub use report::{
    WorkspaceDomainReport, WorkspaceDomainReportEntry, WorkspaceDomainReportInputs,
    WorkspaceDomainResidualReport, WorkspaceDomainStateReport, WorkspaceDomainStoragePopulation,
    WorkspaceDomainTensorBufferReport, WorkspaceReportConstructionError, WorkspaceReportError,
    WorkspaceReportGraph, WorkspaceReportInputs, WorkspaceReportLayout, WorkspaceReportNode,
    WorkspaceReportPlacements, WorkspaceReportResidual, WorkspaceReportScalars,
    WorkspaceReportWorkspace, WorkspaceStoragePopulation,
};
pub use shape::{
    WorkspaceBroadcastShape, WorkspaceMatmulShape, WorkspaceScatterShapeError, WorkspaceShapeError,
    validate_masked_scatter_shapes,
};
mod effects;
pub use effects::{
    WorkspaceEffectError, WorkspaceOutputStorageView, validate_workspace_host_assumptions,
    validate_workspace_output_storage, validate_workspace_tensor_declaration,
};
mod addressable_region;
pub use addressable_region::{
    WorkspaceAddressableChunkPlan, WorkspaceAddressableObservationLayout,
    WorkspaceAddressableObservationSource, WorkspaceAddressableObservationView,
    WorkspaceAddressableRegion, WorkspaceAddressableRegionView, record_addressable_region,
    record_addressable_region_with_observation,
};
mod grouped_observation_envelope;
pub use grouped_observation_envelope::{
    WorkspaceGroupedObservationEnvelope, WorkspaceGroupedObservationExtent,
};
mod expert_region;
pub use expert_region::observation::{
    WorkspaceExpertObservationSource, WorkspaceExpertObservationView,
    WorkspaceGroupedSourceRetention,
};
pub use expert_region::{
    ExpertRegionInputShape, WorkspaceExpertInactiveWave, WorkspaceExpertKernel,
    WorkspaceExpertMovementPopulation, WorkspaceExpertProviderWave, WorkspaceExpertRegion,
    WorkspaceExpertRegionView, WorkspaceExpertTransfer, WorkspaceExpertTransfers,
    record_expert_inactive_wave, record_expert_provider_wave, record_expert_region,
    record_expert_region_with_observation,
};
mod operation;
pub use operation::{
    WorkspaceCollectiveView, WorkspaceLayoutIter, WorkspaceLayoutList, WorkspaceOperationKindView,
    WorkspaceOperationView,
};
mod fact_context;
mod facts;
mod metadata;
mod metadata_funding;
pub(crate) mod policy_clone;
pub use fact_context::WorkspaceMetadataError;
pub use facts::{
    WorkspaceEffectDestination, WorkspaceEffectLayout, WorkspaceFactDestinationError,
    WorkspaceFactDestinationKind, WorkspaceFactMechanisms, WorkspaceHostDestination,
    WorkspaceHostFacts, WorkspaceOperationFacts, WorkspaceOutputEffect,
};
pub use metadata::{
    WorkspaceContextMetadataBuilder, WorkspaceContextMetadataError, WorkspaceMetadataAllocation,
    WorkspaceMetadataEnvelope, visit_isolated_copy_operations,
};
pub use metadata_funding::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
pub use policy_clone::ParameterMetadataAllocation;
mod borrowed_storage;
mod construction;
mod host_value;
mod model_control;
mod tensor;
mod value_completion;
pub use backend::{
    WorkspaceBackend, WorkspaceEmbedding, WorkspaceGroupSelector, WorkspaceGroupedBank,
    WorkspaceGroupedPhase, WorkspaceGroups, WorkspaceLinear, WorkspaceNormalization,
    WorkspaceParallelContext, WorkspaceRotary,
};
pub use backend::{
    WorkspaceBlockwiseAccumulator, WorkspaceBlockwisePolicy, WorkspaceBlockwiseStage,
};
pub use backend::{WorkspaceHyperConnection, WorkspaceHyperHead};
pub use borrowed_storage::{WorkspaceBorrowedStorage, WorkspaceBorrowedStorageError};
pub use construction::WorkspaceImportError;
pub use host_value::{
    WorkspaceHostReadValue, WorkspaceStoredHostIdentity, WorkspaceStoredHostValue,
};
pub use isolated_copy::{
    WorkspaceCopyError, WorkspaceCopyPreparationError, WorkspaceCopyPreparationLayout,
    WorkspaceCopyPreparationLayoutBuilder, WorkspaceIsolatedCopyPlan,
    WorkspaceIsolatedCopyPreparation,
};
pub use model_control::{WorkspaceModelControl, WorkspaceModelControlPhase};
pub use sampling::WorkspaceSamplingOperation;
pub use tensor::{WorkspaceExistingStorage, WorkspaceTensor};

/// Scalar representation used by an equation's metadata tensors.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WorkspaceDtype {
    /// Conservative four-byte floating execution. The mechanism must reject
    /// this descriptor if its selected realization can widen beyond float32.
    Float32,
    /// Four-byte signed integer.
    Int32,
    /// One-byte boolean.
    Bool,
    /// Packed checkpoint bytes, including GGML and FP8 encodings.
    Uint8,
    /// Packed checkpoint words.
    Uint32,
}
impl WorkspaceDtype {
    /// Bytes in one represented scalar.
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Bool | Self::Uint8 => 1,
        }
    }
}

/// Checked tensor geometry, without values or native handles.
#[derive(Debug, Clone)]
pub struct WorkspaceLayout {
    shape: shared_shape::SharedShape,
    dtype: WorkspaceDtype,
    representation: Option<WorkspaceRepresentation>,
}
impl WorkspaceLayout {
    /// Validates all dimensions and storage arithmetic.
    pub fn new(shape: &[i32], dtype: WorkspaceDtype) -> Result<Self, Error> {
        WorkspaceLayoutView::new(shape, dtype).map_err(WorkspaceLayoutError::into_ordinary)?;
        Ok(Self::from_owned_shape(shape.to_vec(), dtype))
    }
    /// Logical dimensions, including any zero extent.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    /// Represented scalar type.
    pub fn dtype(&self) -> WorkspaceDtype {
        self.dtype
    }
    /// Borrows this layout without cloning its shape or allocating metadata.
    pub fn as_view(&self) -> WorkspaceLayoutView<'_> {
        WorkspaceLayoutView {
            shape: &self.shape,
            dtype: self.dtype,
            representation: self.representation,
        }
    }
    /// Logical elements; a rank-zero tensor contains one scalar.
    pub fn elements(&self) -> Result<u64, Error> {
        self.as_view()
            .elements()
            .map_err(WorkspaceLayoutError::into_ordinary)
    }
    /// Logical storage bytes before any selected allocation padding.
    pub fn bytes(&self) -> Result<u64, Error> {
        self.as_view()
            .bytes()
            .map_err(WorkspaceLayoutError::into_ordinary)
    }
}

/// Exact operation identity needed to distinguish native allocation strategies.
#[derive(Debug, Clone)]
pub enum WorkspaceOperationKind {
    /// A complete independently addressable provider, without expert exchange.
    AddressableRegion(Box<WorkspaceAddressableRegion>),
    /// A finite architecture-owned dynamic region. Its exact child source must
    /// be bound from completed IDs/counts; this is not a synthetic static graph.
    ExpertRegion(Box<WorkspaceExpertRegion>),
    ExpertProviderWave(WorkspaceExpertProviderWave),
    ExpertInactiveWave(Box<WorkspaceExpertInactiveWave>),
    /// Constructs an unloaded parameter's scalar seed, before checkpoint
    /// binding replaces its lazy logical value. There are no inputs and one
    /// rank-zero output in the actual placeholder dtype; the logical weight
    /// geometry is not an allocation requested by this operation.
    ParameterPlaceholder,
    /// Initializes a tensor from a scalar or host input.
    Initialize,
    /// Scalar fill or borrowed host initialization in an exact floating type.
    /// Logical storage retains the four-byte envelope; no source credit is granted.
    InitializeFloating(WorkspaceFloatingType),
    /// Actual floating conversion, possibly aliasing when the source type matches.
    /// This is not a view; native facts must include the conversion allocation.
    CastFloating(WorkspaceFloatingType),
    /// Standalone effective checkpoint decoding, distinct from fused packed
    /// matrix multiplication. Native facts must qualify this exact decoder.
    ParameterDecode(crate::parameter_values::ParameterDecoding),
    /// Copy one retained immutable host value into independent device storage,
    /// preserving its exact floating type. The native source and completion
    /// account must be authenticated separately; this descriptor grants none.
    HostTransferFloating(WorkspaceFloatingType),
    /// Stores one Device value into separately admitted Host source storage.
    /// The native completion Array is internal; this effect returns no Tensor.
    HostStoreFloating(WorkspaceStoredHostIdentity, WorkspaceFloatingType),
    /// Loads an exact previously stored Host value into new Device storage.
    /// The source handle is metadata; native source custody remains mandatory.
    HostLoadStoredFloating(WorkspaceStoredHostIdentity, WorkspaceFloatingType),
    /// One fixed Rust F32 initialization buffer followed by the selected
    /// native host-input copy. No inputs and one F32 output; the scalar
    /// expression obeys Tensor::from_f32_fn's allocation-free contract.
    GeneratedF32Initialization,
    /// Sampling primitive selected by the shared runtime sampling policy.
    Sampling(WorkspaceSamplingOperation),
    /// Raw terminal-row candidate extraction, distinct from sampling masks.
    /// One floating [1, rows, vocabulary] input; U32 IDs[count] and F32 scores[count]
    /// outputs. Full-row finite validation and the selected sort workspace are retained.
    CandidateExtraction {
        vocabulary: u32,
        count: u32,
    },
    /// Named elementwise equation. Native facts must explicitly recognize names.
    Elementwise(&'static str),
    /// Metadata transformation; native facts decide whether storage can alias.
    View(&'static str),
    /// Exact normalized axis permutation. The existing paid axis vector is
    /// retained by the trace; this is descriptive, not storage authority.
    Transpose(Vec<usize>),
    /// Static index selection; output axes are already resolved.
    Index {
        /// Integer selectors remove axes, potentially requiring a native
        /// reshape after slicing. Ranges preserve their axes.
        selected_axes: usize,
    },
    /// Exact rank-preserving positive-stride slice. Coordinates are retained
    /// for source qualification; this descriptor grants no native authority.
    StaticSlice {
        /// Inclusive nonnegative start on every axis.
        starts: Vec<i32>,
        /// Exclusive end on every axis.
        ends: Vec<i32>,
        /// Strictly positive stride on every axis.
        strides: Vec<i32>,
    },
    /// Unit-stride replacement of an exact rank-preserving rectangle. Native
    /// facts must include possible full destination copies and update casts;
    /// metadata never assumes that the destination can be donated in place.
    SliceUpdate {
        /// Nonnegative start on every axis; ends follow the update shape.
        starts: Vec<i32>,
    },
    /// Exact positive-stride update without rank changes or update broadcasting.
    /// Native facts include a possible full destination copy and native casts.
    StaticSliceUpdate {
        /// Nonnegative start on every axis.
        starts: Vec<i32>,
        /// Exclusive end on every axis.
        ends: Vec<i32>,
        /// Strictly positive stride on every axis.
        strides: Vec<i32>,
    },
    /// Make row-major storage, possibly aliasing an already contiguous input.
    /// The selected mechanism must cover every stride layout represented here.
    Contiguous,
    /// Independent logical data copy, including selected stride compaction,
    /// synchronization and host/native transfer workspace. This is distinct
    /// from handle cloning.
    DeepCopy,
    /// Native indexed gather, including domain validation.
    Gather {
        /// Validated nonnegative source axis replaced by the index shape.
        axis: usize,
    },
    /// Add rank-two updates into leading-axis rows. Integer indices are
    /// [updates,1]; all non-leading dimensions are exact. Duplicate destinations
    /// accumulate in the target dtype. Actual index-source validation is separate.
    IndexedRowAdd,
    /// Select a rank-one native value source through a rank-one I32 index
    /// source whose in-range values are established by its retained producer.
    /// There is no second native index-validation readout in this worker.
    IndexedElementSelect,
    /// Overwrite rank-one values at producer-validated distinct I32 indices.
    /// Updates have exactly the index shape and the source's physical dtype.
    /// This is general sparse Scatter overwrite, never additive ScatterAxis.
    IndexedElementUpdate,
    /// Concatenation or stacking into the declared output geometry.
    Concatenate,
    /// Dense matrix multiplication with broadcast batch geometry.
    Matmul,
    /// Tensor-level dense affine operator, including optional additive bias.
    DenseLinear,
    /// Affine or packed projection with exact published encoding and companions.
    Projection(LinearFormatSpec),
    /// Observed block-FP8 preparation. Outputs are the actual compact U8
    /// activation and F32 block-scale roots; weight-scale decode stays live.
    ProjectionPrepare(LinearFormatSpec),
    /// Observed block-FP8 projection after its callback. The last two inputs
    /// are exactly the compact activation/scale outputs of preparation.
    ProjectionFinish(LinearFormatSpec),
    /// Fixed reconstruction of compact quantizer-produced U8 E4M3 values to
    /// F32. This is not a general arbitrary-stride conversion contract.
    BlockFp8ActivationDecode,
    /// Learned routing with the complete scoring and precision policy. Optional
    /// supplied IDs and intervention controls retain their distinct execution.
    GroupSelection {
        /// Exact construction and physical parameter topology.
        spec: Box<crate::TopKGroupSelectorSpec>,
        /// Caller supplied the selected IDs instead of requesting top-k.
        supplied_indices: bool,
        /// Validated pre-dispatch control, including original-decision capture.
        control: Option<Box<crate::routing_intervention::GroupSelectionControl>>,
    },
    /// Joint selected and always-on coefficient normalization.
    JointGroupSelection(crate::JointGroupSelectionSpec),
    /// Selected grouped computation. The mechanism must bound every route
    /// distribution consistent with these shapes, including sorting and staging.
    Grouped {
        /// Retained architecture equation and exact parameter organization.
        bank: Box<WorkspaceGroupedBank>,
        /// Whole equation or the two sides of an observed unit boundary.
        phase: WorkspaceGroupedPhase,
        /// Rank-local partial; replicated output bias is a separate result.
        partitions: Option<usize>,
    },
    /// FP32 residual preparation, learned mixing, Sinkhorn passes and collapse.
    HyperCollapse(Box<crate::HyperConnectionSpec>, f32),
    /// Sublayer injection and residual mixing using retained coefficients.
    HyperExpand,
    /// Learned final stream coefficients, before an optional observer.
    HyperHeadCoefficients(Box<crate::HyperHeadSpec>),
    /// Final weighted stream sum consuming the observed coefficients.
    HyperHeadSum,
    /// Row-parallel affine operation, including reduction and native bias order.
    RowParallelProjection(LinearFormatSpec, usize),
    /// Embedding lookup with exact encoding and index validation/sentinel policy.
    Embedding(LinearFormatSpec, crate::EmbeddingLookupPolicy),
    /// Global token validation, local row selection and zeroing contributions
    /// outside one rank's ownership. The following sum is traced separately.
    VocabularyParallelLookup {
        /// Exact local parameter encoding and companions.
        format: LinearFormatSpec,
        /// Architecture-declared global and local vocabulary coordinates.
        range: crate::VocabularyParallelRange,
        /// Global validation and sentinel policy.
        policy: crate::EmbeddingLookupPolicy,
    },
    /// Reduction over one validated axis, retaining numerical precision policy.
    Reduction(&'static str, i32, bool),
    /// Final-axis RMS or related normalization, including optional grouping.
    Normalization(&'static str, Option<i32>),
    /// Layer normalization with the exact optional affine operand roles.
    LayerNorm {
        /// A learned multiplicative weight follows the input.
        weight: bool,
        /// An additive bias follows the input and optional weight.
        bias: bool,
    },
    /// Constructed normalization, including learned-offset and grouping policy.
    ConstructedNormalization(crate::NormalizationConstructionSpec),
    /// Fused gated product with exact clipping and activation policy.
    GatedProduct(crate::GatedProductPolicy),
    /// Gated-delta recurrence; outputs are the final FP32 matrix and sequence.
    GatedDeltaScan,
    /// Selective state-space recurrence, retaining native chunk policy and floor.
    SelectiveStateSpaceScan(usize, f32),
    /// Rotary transformation with its complete architecture-declared algorithm.
    Rotary(crate::RotarySpec, Option<i32>),
    /// Explicit reciprocal-frequency rotary transformation.
    RotaryFrequencies(i32, bool, i32),
    /// Tensor-level rotary recipe (dimensions, adjacent pairs, base, scale, offset).
    TensorRotary(i32, bool, f32, f32, i32),
    /// Explicit multi-axis position frequencies and their cosine/sine outputs.
    MultiAxisRotary(crate::multimodal::MultiAxisRotarySpec),
    /// The same tensor operations with immutable caller-supplied host frequencies.
    /// Metadata is ordinary tracing storage; this carries no source-account credit.
    PreparedMultiAxisRotary(crate::multimodal::MultiAxisRotarySpec),
    /// Centroid selection, selected vocabulary projection and per-position fill.
    MaskedOutputProjection {
        /// Number of centroid groups projected for each position.
        top_centroids: i32,
        /// Positive offset below each position's selected minimum.
        mask_margin: f32,
    },
    /// Shared normalization over local and selected pooled keys.
    IndexedAttention {
        /// Query/key score multiplier.
        scale: f32,
        /// Local eligibility mask is present.
        local_mask: bool,
        /// Pooled eligibility mask is present.
        pooled_mask: bool,
        /// Learned normalization sinks are present.
        sinks: bool,
    },
    /// Shared normalization over local and complete pooled keys.
    PooledAttention {
        /// Query/key score multiplier.
        scale: f32,
        /// Local eligibility mask is present.
        local_mask: bool,
        /// Pooled eligibility mask is present.
        pooled_mask: bool,
        /// Learned normalization sinks are present.
        sinks: bool,
    },
    /// Weighted head-score reduction and selection of pooled source positions.
    PooledPositions {
        /// Maximum selected pooled positions per query.
        top_k: i32,
        /// Query/key score multiplier.
        scale: f32,
        /// Multiplier applied to head weights.
        head_scale: f32,
        /// Pooled eligibility mask is present.
        masked: bool,
    },
    /// Gather query-specific pooled eligibility at selected positions.
    GatherPooledMask,
    /// Relative-profile attention with absolute positions, window and log scaling.
    RelativeAttention {
        /// Absolute first query position.
        query_offset: i32,
        /// Absolute first retained key position.
        key_offset: i32,
        /// Optional causal window.
        window: Option<i32>,
        /// Optional floor for logarithmic scaling.
        log_scaling_floor: Option<i32>,
        /// Logarithmic scale multiplier.
        log_scaling_alpha: f32,
    },
    /// Attention; causal/sliding and numerical policy are explicit.
    Attention {
        /// Implicit causal mask.
        causal: bool,
        /// Optional token-count window and absolute first query position.
        window: Option<(i32, i32)>,
        /// Learned normalization sinks are present.
        sinks: bool,
        /// Attention score soft-capping is present.
        softcap: bool,
        /// Selected attention arithmetic.
        arithmetic: crate::AttentionArithmetic,
    },
    /// The ordinary online-softmax interface over ordered reconstructed cache
    /// blocks. Per-block transient facts remain separate from cache residency.
    BlockwiseAttention {
        /// Absolute causal, window, prefix and mask coordinates.
        policy: WorkspaceBlockwisePolicy,
        /// Initial preparation, recurrence update or final normalization.
        stage: WorkspaceBlockwiseStage,
    },
    /// Completes exactly the supplied value roots before the next mechanism
    /// phase. No new tensor storage or output value is constructed. This is a
    /// trace descriptor, not native submission or source authority.
    ValueCompletion,
    /// Completes local roots through the selected communication backend's
    /// ordinary operation submission. This describes the shared driver worker;
    /// it grants no native source, account, or submission authority.
    CommunicationDependencies,
    /// One actual model-internal control call reached by the shared partition driver.
    /// This source descriptor grants no native authority or tensor storage.
    CommunicationControl(WorkspaceModelControl),
    /// Completes the two actual paged backing roots before canonical block
    /// publication. Native manager/source validation remains independently
    /// required; this descriptor grants no publication authority.
    CachePublicationCompletion,
    /// Completes the paged scan result before releasing its source leases.
    /// Source pins, manager metadata and native completion remain backend-owned.
    CacheScanCompletion,
    /// Keeps actual existing values in the enclosing completion root union.
    ValueRetention,
    /// Boolean mask with prompt and cached-key geometry.
    CausalMask(crate::operation_geometry::CausalMaskGeometry),
    /// Eligibility of complete pooled source windows at exact query positions.
    PoolingMask(crate::operation_geometry::PoolingMaskGeometry),
    /// Convolution dimensions and normalized geometry.
    Convolution {
        /// One entry per spatial axis.
        stride: Vec<i32>,
        /// Symmetric padding per spatial axis.
        padding: Vec<i32>,
        /// Dilation per spatial axis.
        dilation: Vec<i32>,
        /// Convolution groups.
        groups: i32,
        /// Output padding for transposed convolution; absent for forward.
        transposed: Option<Vec<i32>>,
    },
    /// Padding with the declared boundary rule.
    Pad(crate::PadMode),
    /// One rank's collective with exact peer geometry. Transport scratch must
    /// still be priced by the retained native mechanism.
    Collective(WorkspaceCollective),
}

/// Geometry of a collective in one rank's ordinary neural equation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum WorkspaceCollective {
    /// One selected canonical boundary transfer. Native lowering supplies the
    /// byte equations, exact route source, backing and completion populations.
    Boundary {
        /// Opaque runtime route identity; this scalar grants no source authority.
        route: u64,
        /// Role position in the canonical schema.
        ordinal: usize,
        /// Actual canonical in-band header length.
        header_bytes: usize,
    },
    /// Sum same-shaped contributions from all peers.
    Sum {
        /// Participating ranks.
        partitions: usize,
        /// Rank whose storage is being traced.
        rank: usize,
    },
    /// Sum the owner's observed contribution for the selected output publication.
    /// The caller separately traces the ordinary non-owner zero multiplication.
    Broadcast {
        /// Exact retained publication group, distinct from neural TP selection.
        group: eredu_core::CollectiveGroupId,
        /// Owner's local index in the selected group.
        root: usize,
        /// Exact selected group population.
        partitions: usize,
        /// Local index of this rank in the selected group.
        rank: usize,
    },
    /// Native first-axis Gather of equally padded shards within the shared
    /// uneven-axis wrapper. Output is still in native rank-block order.
    GatherFirstAxis {
        /// Axis padded by the enclosing wrapper before native communication.
        axis: usize,
        /// Local rank whose original shard is represented.
        rank: usize,
        /// Original unpadded peer widths; native input uses their maximum.
        peer_widths: Vec<usize>,
    },
    /// Gather contiguous shards into the selected logical axis.
    Gather {
        /// Nonnegative concatenation axis.
        axis: usize,
        /// Rank whose local shard is the operation input.
        rank: usize,
        /// Exact per-peer axis widths, including uneven partitions.
        peer_widths: Vec<usize>,
    },
}

/// One invocation of an existing neural equation, with no native values.
#[derive(Debug, Clone)]
pub struct WorkspaceOperation {
    /// Primitive and all allocation-relevant semantic options.
    pub kind: WorkspaceOperationKind,
    /// Exact logical inputs in operation order.
    pub inputs: Vec<WorkspaceLayout>,
    /// Exact logical outputs in operation order.
    pub outputs: Vec<WorkspaceLayout>,
}

/// Storage effect of one result. Cloning a handle never duplicates its charge.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum WorkspaceOutputStorage {
    /// An independently allocated result, including allocation padding.
    Allocate(u64),
    /// Shares the complete backing allocation of the indicated input.
    AliasInput(usize),
    /// Shares an earlier output's complete backing, including its possible
    /// input aliases. Only backward references are valid; no allocation or
    /// charge is created for another logical view of that output.
    AliasOutput(usize),
    /// The selected primitive may allocate or reuse any listed input. The
    /// conservative trace charges the possible new allocation and retains every
    /// candidate backing root. This does not assume native donation or strides.
    AllocateOrAliasInputs {
        /// Bound on a possible independent output allocation.
        bytes: u64,
        /// Possible aliased inputs, indexed in operation input order.
        inputs: Vec<usize>,
    },
}

/// A proved tensor-buffer bound, excluding already-retained inputs. Separate
/// host workspace must be supplied through `WorkspaceMechanisms::host_workspace_bound`.
#[derive(Debug, Clone)]
pub struct WorkspaceOperationBound {
    /// Exactly one storage effect per output.
    pub outputs: Vec<WorkspaceOutputStorage>,
    /// Temporary tensor buffers beyond independently allocated outputs.
    pub scratch_bytes: u64,
    /// Derivation and selected implementation assumptions; never telemetry.
    pub assumptions: String,
}

/// Managed host workspace of one operation, disjoint from its tensor buffers.
/// This includes operation-owned staging, conversion, sorting and transfer
/// buffers. Host mappings of already-counted tensor storage are not additional
/// allocations. Existing parameter/state residency and request preparation are
/// separate caller responsibilities. Runtime/driver bookkeeping, allocator
/// caches, JIT programs and unrelated process memory are outside this contract.
#[derive(Debug, Clone)]
pub struct WorkspaceHostBound {
    /// Maximum additional host bytes retained through the completed span.
    pub bytes: u64,
    /// Selected implementation and lifetime derivation; required even for zero.
    pub assumptions: String,
}

/// The selected mechanism's completion boundary. Both choices traverse shared
/// drivers and require independent native source and completion facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceCompletionStrategy {
    /// Each operation owns its submission and completion resources.
    OperationSubmissions,
    /// Model operations borrow the enclosing admitted submission and settle
    /// their dependencies through its nested completion mechanism.
    EnclosingSubmission,
}

/// Actual operation-local backing populations produced by an allocating cold
/// mechanism. The shared trace consumes these rows through the same lifetime
/// reducer as lexical fact sources. This is metadata, never execution authority.
#[derive(Debug)]
pub struct WorkspaceOperationAllocationSources {
    /// Simultaneous scratch rows, or pre-resolved complete source alternatives.
    /// Its backing-byte envelope must equal the operation's scratch fact.
    pub scratch: Option<WorkspaceAllocationPopulation>,
    /// Independent output sources in result order; absent rows use the ordinary
    /// effect and placement. Aliases must not carry an independent source.
    pub outputs: Vec<Option<WorkspaceAllocationPopulation>>,
}

/// Side-effect-free native allocation facts. Unknown primitives return `None`;
/// errors describe invalid geometry or arithmetic rather than filling a gap.
/// Bounds must cover every numerical value and stride layout consistent with
/// the descriptor, including widening, contiguous copies and host staging.
/// Tensor buffers and disjoint managed host workspace are separate mandatory
/// contributions, even when they share physical memory. Pricing one never
/// supplies a missing fact for the other.
/// Metadata does not assume values, contiguity or in-place donation. A mechanism
/// may only declare an alias when it is valid for all those possibilities.
pub trait WorkspaceMechanisms: fmt::Debug {
    /// Prepare operation-local source populations for an ordinary cold context.
    /// Implementations fund every owning constructor through `context` and use
    /// the same selected source worker as their lexical fact implementation.
    /// Returning `None` preserves the preowned source loans below. Finite fact
    /// contexts use `WorkspaceFactMechanisms::with_prepared_facts` instead.
    fn prepare_allocation_sources(
        &self,
        _operation: WorkspaceOperationView<'_>,
        _context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceOperationAllocationSources>, Error> {
        Ok(None)
    }

    /// Descriptive scheduling input for shared cache and communication traversal.
    /// The default follows their ordinary per-operation submission paths.
    /// Selecting an enclosing submission never creates execution authority.
    fn completion_strategy(&self) -> WorkspaceCompletionStrategy {
        WorkspaceCompletionStrategy::OperationSubmissions
    }
    /// Immutable physical topology established by this selected mechanism.
    /// Missing facts cannot grant physical-domain allocation permission.
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        None
    }

    /// Placement of the independent backing, when this output allocates.
    /// Aliases preserve the original backing's placement instead.
    fn output_placement(
        &self,
        _operation: WorkspaceOperationView<'_>,
        _output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        None
    }

    /// Placement of this operation's native scratch allocations.
    fn scratch_placement(
        &self,
        _operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        None
    }
    /// Source-owned scratch allocations retained through this operation's completion.
    /// When present, these exact descriptors replace the homogeneous scratch
    /// placement. Their checked byte sum must equal the operation's scratch fact.
    /// Distinct occurrences are independent populations; this loan supplies no
    /// physical owner, storage credit, or execution authority.
    fn scratch_allocations(
        &self,
        _operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        Ok(None)
    }
    /// Actual simultaneous backing rows retained by one new output. The raw
    /// source rows replace a homogeneous output placement and keep separate
    /// storage identities under the existing alias/lifetime graph. Resolved
    /// alternative requirement groups cannot substitute for backing identities.
    fn output_allocations(
        &self,
        _operation: WorkspaceOperationView<'_>,
        _output: usize,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Error> {
        Ok(None)
    }
    /// Maximum independent backings in this operation's temporary allocation
    /// source. This is independent of host-control bytes and cannot be inferred
    /// from a different allocator or a scalar memory allowance.
    fn scratch_allocation_count(
        &self,
        _operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<usize>, Error> {
        Ok(None)
    }
    /// Host controls owned by one newly allocated backing for its full lifetime.
    /// Aliases reuse their backing's controls. `None` makes attribution incomplete.
    fn allocation_host_control_bytes(
        &self,
        _operation: WorkspaceOperationView<'_>,
        _output: usize,
    ) -> Option<u64> {
        Some(0)
    }
    /// Host controls retained by all temporary native scratch allocations.
    fn scratch_host_control_bytes(
        &self,
        _operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        Ok(Some(0))
    }

    /// Prospective Units layout from the retained addressable member sources.
    /// Absence carries no floating witness and grants no numerical work.
    fn addressable_observation_layout(
        &self,
        _source: WorkspaceAddressableRegionView<'_>,
        _inputs: &[WorkspaceLayout],
        _context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceAddressableObservationLayout>, Error> {
        Ok(None)
    }

    /// Integer element type of this selected worker's ordinary prepared text
    /// token source. This descriptive fact creates no tensor, readback or
    /// allocation permission. Explicitly borrowed inputs retain their own dtype.
    fn prepared_text_input_dtype(&self) -> Option<WorkspaceDtype> {
        None
    }

    /// Exact physical representation of a successful output of this selected
    /// worker. Absence preserves the full dtype/stride union. Implementations
    /// must use only actual input evidence and the primitive's own semantics;
    /// this supplies neither existing-storage credit nor execution authority.
    fn output_representation(
        &self,
        _operation: WorkspaceOperationView<'_>,
        _output: usize,
    ) -> Option<WorkspaceRepresentation> {
        None
    }

    /// Selects actual ordinary projection-input provenance. The existing
    /// identity-input formats preserve their borrowed callback contract;
    /// transformed FP8 inputs require an explicit provider mechanism.
    fn projection_input_observation_mechanism(
        &self,
        format: &LinearFormatSpec,
    ) -> Result<Option<crate::ProjectionInputObservationMechanism>, Error> {
        format.validate()?;
        Ok(
            if matches!(
                format.encoding(),
                eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
            ) {
                None
            } else {
                Some(crate::ProjectionInputObservationMechanism::Borrowed)
            },
        )
    }

    /// Exact callback partition selected for a grouped unit boundary. An absent
    /// fact cannot be replaced by a whole-batch callback: callback allocations,
    /// cumulative budgets and coordinates depend on this schedule.
    fn grouped_observation_schedule(
        &self,
        _bank: &WorkspaceGroupedBank,
        _tokens: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        Ok(None)
    }

    /// Prices output storage and scratch for this exact invocation.
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error>;

    /// Prices managed host workspace for the same invocation. Absence is
    /// unknown, including for tensor aliases and views. A proved zero requires
    /// an explicit bound; a tensor-buffer fact alone cannot establish it.
    fn host_workspace_bound(
        &self,
        _operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(None)
    }
}

/// Data-independent token partition of a selected grouped observer mechanism.
/// An empty input delivers one empty batch. Chunks remain in source-token order,
/// with a final short batch; selected rows are sorted separately in each batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceGroupedObservationSchedule {
    /// One callback covers the entire logical invocation.
    WholeBatch,
    /// Each callback covers at most this many source tokens.
    TokenChunks(std::num::NonZeroU32),
}

/// A grouped observation quote cannot run without its selected delivery schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("selected grouped observation schedule is unavailable for workspace tracing")]
pub struct WorkspaceGroupedObservationScheduleUnavailable;

#[derive(Debug)]
struct Storage {
    bytes: Option<u64>,
    host_control_bytes: Option<u64>,
    placement: Option<eredu_core::MemoryPlacement>,
    maximum_allocations: usize,
    population: Option<WorkspaceAllocationPopulation>,
    possible_aliases: Vec<Rc<Storage>>,
}
#[derive(Debug, Default)]
struct Trace {
    report_finished: bool,
    opening_state: Option<Vec<Rc<Storage>>>,
    allocations: Vec<Rc<Storage>>,
    scratch: u64,
    scratch_overflow: bool,
    placed_scratch: Vec<WorkspaceScratchAllocation>,
    scratch_sources: Vec<WorkspaceAllocationPopulation>,
    host_workspace: u64,
    host_staging_incomplete: bool,
    operations: Vec<WorkspaceOperation>,
    missing: Vec<usize>,
    missing_host: Vec<usize>,
    assumptions: Vec<String>,
}

// Separate from the trace loan: cloning a value may happen while a shared
// equation visitor already holds that loan. Overflow stays unknown until reset.
#[derive(Debug)]
struct WorkspaceIdentity {
    tensor_handle_clones: Cell<Option<usize>>,
    fact_bytes: Cell<Option<usize>>,
    fact_remaining: Cell<usize>,
    metadata_bytes: Cell<Option<usize>>,
    metadata_reports: Cell<Option<usize>>,
    metadata_remaining: Cell<usize>,
}
impl WorkspaceIdentity {
    fn new() -> Self {
        Self {
            tensor_handle_clones: Cell::new(Some(0)),
            fact_bytes: Cell::new(None),
            fact_remaining: Cell::new(usize::MAX),
            metadata_bytes: Cell::new(None),
            metadata_reports: Cell::new(None),
            metadata_remaining: Cell::new(usize::MAX),
        }
    }
    fn record_tensor_clone(&self) {
        self.tensor_handle_clones.set(
            self.tensor_handle_clones
                .get()
                .and_then(|n| n.checked_add(1)),
        );
    }
}

/// Cold metadata context. Each context is one selected mechanism and trace;
/// values from different contexts cannot be combined accidentally. Clones share
/// the same ledger, so independently scheduled metadata lanes remain charged
/// together until the caller explicitly begins the next completed span.
#[derive(Clone)]
pub struct WorkspaceContext {
    identity: Rc<WorkspaceIdentity>,
    mechanisms: Rc<dyn WorkspaceMechanisms>,
    facts: Option<Rc<dyn facts::owned::FiniteWorkspaceFacts>>,
    trace: Rc<RefCell<Trace>>,
    borrowed: Rc<RefCell<Option<WorkspaceBorrowedStorage>>>,
    parameter_representations: Rc<RefCell<Option<Vec<WorkspaceParameterRepresentation>>>>,
    tracing_started: Rc<report::Lifecycle>,
    // Last: all Context-owned shells retire before this accounting alias.
    funding: Option<HostMetadataFunding>,
}
impl fmt::Debug for WorkspaceContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkspaceContext")
            .field("mechanisms", &self.mechanisms)
            .finish_non_exhaustive()
    }
}

/// Conservative storage demand for a completed equation span. Native facts must
/// cover every operation before any byte total may authorize strict execution.
#[derive(Debug, Clone)]
pub struct WorkspaceTraceReport {
    /// Per-domain results from these same original allocation identities.
    /// Absence is missing physical attribution, independent of limit mode.
    pub physical_domains: Option<WorkspaceDomainReport>,
    /// Complete backing union of the supplied closing roots. This stays separate
    /// from the ordinary retained/transient equation totals and native lifetime.
    pub closing_storage: WorkspaceStoragePopulation,
    /// Complete seeded opening backing; absent seeding is not an empty source.
    pub opening_storage: Option<WorkspaceStoragePopulation>,
    /// Optional exact-root residual diagnostic. It never changes the full
    /// report and grants no native storage proof or allocation permission.
    pub residual: Option<WorkspaceResidualReport>,
    /// Inference accounting seeded with exact state roots before the span.
    /// Absent for primitive-only traces, which cannot authorize inference.
    pub state: Option<WorkspaceStateSpanReport>,
    /// All unique tensor buffers and managed host workspace. Unknown if either
    /// required domain is unpriced. This is not a total-process memory bound.
    pub total_bytes: Option<u64>,
    /// New tensor backing storage retained as persistent state by the caller.
    /// Non-tensor persistent state is priced outside this trace.
    pub retained_bytes: Option<u64>,
    /// Tensor and managed host storage beyond those retained state allocations.
    pub transient_bytes: Option<u64>,
    /// Separately inspectable tensor-buffer bounds. These may be known even
    /// when the complete managed-workspace bound is unknown.
    pub tensor_buffers: WorkspaceTensorBufferReport,
    /// Disjoint operation-owned host workspace, or an explicit coverage gap.
    pub host_workspace_bytes: Option<u64>,
    /// Tensor-handle clones executed in this span, including clones of existing
    /// state and source values. This counts aliases, never extra backing storage;
    /// a native realization supplies its own handle-storage cost. None means the
    /// checked counter overflowed. Reporting or cloning a report adds no count.
    pub tensor_handle_clones: Option<usize>,
    /// Actual fact output/alias/text and typed-error construction bytes in this
    /// span. This excludes operation/shape/state/report storage and is not a
    /// total metadata bound. Absent for legacy allocating fact providers.
    pub fact_construction_bytes: Option<usize>,
    /// Participating metadata producer census; never native execution authority.
    pub metadata_construction: Option<WorkspaceMetadataEnvelope>,
    /// Every executed primitive, including view operations and repeated calls.
    pub operations: Vec<WorkspaceOperation>,
    /// Indices into `operations` for primitives without tensor-buffer bounds.
    pub unpriced_operations: Vec<usize>,
    /// Indices into `operations` without managed host-workspace bounds or
    /// native scratch allocation control costs. Known staging bytes remain
    /// separately available in `host_workspace_bytes`.
    pub unpriced_host_operations: Vec<usize>,
    /// Unique selected-mechanism assumptions used by known operations.
    pub assumptions: Vec<String>,
}

/// Incremental demand excluding only explicitly selected existing identities.
/// The complete opening/closing/new union is traversed before classification;
/// borrowed capacities are never subtracted from an already-computed maximum.
#[derive(Debug, Clone)]
pub struct WorkspaceResidualReport {
    /// Exact unborrowed opening backing population; absent without state seeding.
    pub opening_storage: Option<WorkspaceStoragePopulation>,
    /// Exact unborrowed closing backing population; no scalar source subtraction.
    pub closing_storage: Option<WorkspaceStoragePopulation>,
    /// Immutable context-bound selection shared by every span of this trace.
    pub borrowed_storage: WorkspaceBorrowedStorage,
    /// All unborrowed opening, closing and new backing, plus scratch and host
    /// workspace. Unknown without seeded state or complete operation coverage.
    pub total_bytes: Option<u64>,
    /// Unborrowed closing roots, including unchanged existing components.
    pub retained_bytes: Option<u64>,
    /// Unborrowed opening roots absent from the closing reachability set.
    pub displaced_bytes: Option<u64>,
    /// Unborrowed union roots absent from closing state, plus scratch and host.
    pub transient_bytes: Option<u64>,
}

/// Complete retained-state ownership across one completed equation span.
/// Roots from before the span keep their allocation identities and capacities;
/// replacing state does not release its old storage before native completion.
#[derive(Debug, Clone)]
pub struct WorkspaceStateSpanReport {
    /// All unique closing state allocations, including older backing storage.
    pub retained_bytes: Option<u64>,
    /// Opening state backing no longer reachable from the closing state. This
    /// is transient overlap, even when no new operation allocates that storage.
    pub displaced_bytes: Option<u64>,
    /// New transient allocations/scratch/host storage plus displaced state.
    pub transient_bytes: Option<u64>,
}

impl WorkspaceTraceReport {
    /// Complete inference transient bound, including displaced opening state.
    /// Requires both state seeding and complete closing backing bounds.
    pub fn inference_transient_bytes(&self) -> Option<u64> {
        let state = self.state.as_ref()?;
        state.retained_bytes?;
        state.transient_bytes
    }
}

/// Active tensor-buffer demand within a trace. Allocator observations can be
/// compared with this domain without treating it as a complete admission quote.
#[derive(Debug, Clone)]
pub struct WorkspaceTensorBufferReport {
    /// All unique new tensor buffers and temporary tensor scratch.
    pub total_bytes: Option<u64>,
    /// New tensor backing storage retained by the caller as persistent state.
    pub retained_bytes: Option<u64>,
    /// Tensor-buffer storage beyond those retained state roots.
    pub transient_bytes: Option<u64>,
}

impl WorkspaceContext {
    /// Retains cold native facts. Construction allocates only host metadata.
    pub fn new(mechanisms: impl WorkspaceMechanisms + 'static) -> Self {
        Self {
            identity: Rc::new(WorkspaceIdentity::new()),
            mechanisms: Rc::new(mechanisms),
            facts: None,
            trace: Rc::new(RefCell::new(Trace::default())),
            borrowed: Rc::new(RefCell::new(None)),
            parameter_representations: Rc::new(RefCell::new(None)),
            tracing_started: Rc::new(report::Lifecycle::new()),
            funding: None,
        }
    }

    /// Installs one immutable borrowed-root selection before the first span or
    /// successful operation. Context clones share it and span resets preserve
    /// it. This is metadata classification, not authority to reuse native bytes.
    pub fn set_borrowed_storage(&self, borrowed: WorkspaceBorrowedStorage) -> Result<(), Error> {
        self.set_borrowed_storage_checked(borrowed)
            .map_err(WorkspaceBorrowedStorageError::into_ordinary)
    }

    /// The same one-time selection transition with a fixed, nonallocating cause.
    /// Caller separately retains and funds the metadata selection; success gives
    /// no native storage or submission permission.
    pub fn set_borrowed_storage_checked(
        &self,
        borrowed: WorkspaceBorrowedStorage,
    ) -> Result<(), WorkspaceBorrowedStorageError> {
        if !borrowed.belongs_to(self) {
            return Err(WorkspaceBorrowedStorageError::SelectionContext);
        }
        let mut selected = self.borrowed.borrow_mut();
        if self.tracing_started.get() || selected.is_some() {
            return Err(WorkspaceBorrowedStorageError::SelectionStarted);
        }
        *selected = Some(borrowed);
        Ok(())
    }
    /// Shares the exact installed metadata selection, including an explicit
    /// empty selection. This grants neither physical storage credit nor work
    /// authority; those require the independently retained registered owner.
    pub fn borrowed_storage_selection(&self) -> Option<WorkspaceBorrowedStorage> {
        self.borrowed.borrow().clone()
    }
    /// Borrows the actual ordinary prepared-text producer's integer dtype.
    /// An absent fact leaves portable caller-selected input policy unchanged.
    pub fn prepared_text_input_dtype(&self) -> Option<WorkspaceDtype> {
        self.mechanisms.prepared_text_input_dtype()
    }

    /// Borrows the selected provider's exact input-observation mechanism.
    /// This is metadata provenance, not source registration or work authority.
    pub fn projection_input_observation_mechanism(
        &self,
        format: &LinearFormatSpec,
    ) -> Result<Option<crate::ProjectionInputObservationMechanism>, Error> {
        self.mechanisms
            .projection_input_observation_mechanism(format)
    }

    /// Whether two contexts share the original metadata equation ledger.
    pub fn shares_trace(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.identity, &other.identity)
    }

    /// Validates submission dependencies without executing values, resolving a
    /// quote, or clearing any accumulated allocation charges.
    pub fn validate_values<'a>(
        &self,
        values: impl IntoIterator<Item = &'a WorkspaceTensor>,
    ) -> Result<(), Error> {
        for value in values {
            value.validate_context(self)?;
        }
        Ok(())
    }
    /// Begins a new span after module and existing-state construction. Existing
    /// tensors keep their identities but their storage is not charged again.
    pub fn begin_span(&self) {
        self.tracing_started.set(true);
        *self.trace.borrow_mut() = Trace::default();
        self.identity.tensor_handle_clones.set(Some(0));
    }

    /// Begins an inference span with its exact retained state, after previous
    /// native work would have completed. Empty roots explicitly describe empty
    /// state. Existing weights and non-state resources remain separately priced.
    /// A foreign-context value rejects the transition without clearing charges.
    pub fn begin_state_span<'a>(
        &self,
        retained: impl IntoIterator<Item = &'a WorkspaceTensor>,
    ) -> Result<(), Error> {
        let mut opening_state = Vec::new();
        for value in retained {
            value.validate_context(self)?;
            self.reserve_metadata_vec(&mut opening_state, 1)?;
            opening_state.push(value.storage.clone());
        }
        self.tracing_started.set(true);
        *self.trace.borrow_mut() = Trace {
            opening_state: Some(opening_state),
            ..Trace::default()
        };
        self.identity.tensor_handle_clones.set(Some(0));

        Ok(())
    }

    /// Number of operations already emitted in this current cold span. A
    /// construction driver may retain this ordinal around its own fixed input
    /// producer; it is neither a native identity nor a memory allowance.
    pub fn operation_count(&self) -> usize {
        self.trace.borrow().operations.len()
    }

    /// Completion strategy supplied by this context's selected mechanism.
    pub fn completion_strategy(&self) -> WorkspaceCompletionStrategy {
        self.mechanisms.completion_strategy()
    }

    /// Reports the span, assigning explicitly retained roots to persistent state
    /// once even when several cache views refer to the same allocation.
    pub fn report(&self, retained: &[WorkspaceTensor]) -> Result<WorkspaceTraceReport, Error> {
        self.fixed_report(retained)
    }

    /// Records one declared primitive in this context. This creates metadata
    /// values only; a missing native tensor or host bound remains unknown.
    /// Portable mechanisms above neural contracts use the same ledger instead
    /// of inventing independent allocation accounting.
    pub fn execute(
        &self,
        kind: WorkspaceOperationKind,
        inputs: &[&WorkspaceTensor],
        outputs: Vec<WorkspaceLayout>,
    ) -> Result<Vec<WorkspaceTensor>, Error> {
        if self.trace.borrow().report_finished {
            return Err(WorkspaceMetadataError::ReportFinished.into());
        }
        for input in inputs {
            input.validate_context(self)?;
        }
        let mut input_layouts = self.metadata_vec(inputs.len())?;
        input_layouts.extend(inputs.iter().map(|input| input.layout.clone()));
        let mut operation = WorkspaceOperation {
            kind,
            inputs: input_layouts,
            outputs,
        };
        self.charge_metadata(
            representation::operation_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        // Output evidence belongs to the selected primitive, not to a logical
        // Float32 layout or an earlier unrelated tensor with the same shape.
        for index in 0..operation.outputs.len() {
            let representation = self
                .mechanisms
                .output_representation(operation.as_view(), index);
            let representation = if operation.outputs[index].dtype == WorkspaceDtype::Float32 {
                representation
            } else {
                None
            };
            operation.outputs[index].representation = representation;
        }
        let (mut bound, host) = self.emit_operation_facts(&operation)?;
        let mut sources = if self.facts.is_none() {
            self.charge_metadata(std::mem::size_of::<
                Option<WorkspaceOperationAllocationSources>,
            >())?;
            self.mechanisms
                .prepare_allocation_sources(operation.as_view(), self)?
        } else {
            None
        };
        if sources
            .as_ref()
            .is_some_and(|s| s.outputs.len() > operation.outputs.len())
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut trace = self.trace.borrow_mut();
        let scratch = bound.as_ref().map_or(0, |bound| bound.scratch_bytes());
        let next_scratch = trace.scratch.checked_add(scratch);
        if next_scratch.is_none() && self.mechanisms.memory_topology().is_none() {
            return Err(workspace_overflow("workspace scratch sum overflow"));
        }
        let mut missing_scratch_controls = false;
        let explicit = bound
            .as_mut()
            .and_then(|value| value.take_scratch_allocations());
        let prepared = sources
            .as_mut()
            .and_then(|s| s.scratch.take())
            .map(|source| {
                self.copy_scratch_population(&source, scratch)
                    .map(|rows| (rows, source))
            })
            .transpose()?;
        let explicit = match explicit.or(prepared) {
            Some(rows) => Some(rows),
            None if self.facts.is_none() => self
                .mechanisms
                .scratch_allocations(operation.as_view())?
                .map(|source| {
                    self.copy_scratch_population(source, scratch)
                        .map(|rows| (rows, source.clone()))
                })
                .transpose()?,
            None => None,
        };
        if let Some((rows, source)) = explicit {
            source.validate_domains(self.memory_topology().ok_or_else(|| {
                Error::backend_retained_source(WorkspacePlacementError::MissingTopology)
            })?)?;
            missing_scratch_controls |= source.host_control_bytes().is_none();
            for row in &rows {
                if let Some(placement) = &row.placement {
                    placement
                        .validate(
                            self.memory_topology()
                                .ok_or(WorkspacePlacementError::MissingTopology)
                                .map_err(Error::backend_retained_source)?,
                        )
                        .map_err(Error::backend_retained_source)?;
                }
                missing_scratch_controls |= row.host_control_bytes.is_none();
            }
            self.reserve_metadata_vec(&mut trace.scratch_sources, 1)?;
            self.reserve_metadata_vec(&mut trace.placed_scratch, rows.len())?;
            trace.scratch_sources.push(source);
            trace.placed_scratch.extend(rows);
        } else if scratch != 0 {
            let placement =
                self.copy_placement(self.mechanisms.scratch_placement(operation.as_view()))?;
            self.reserve_metadata_vec(&mut trace.placed_scratch, 1)?;
            let host_control_bytes = self
                .mechanisms
                .scratch_host_control_bytes(operation.as_view())?;
            missing_scratch_controls = host_control_bytes.is_none();
            trace.placed_scratch.push(WorkspaceScratchAllocation {
                bytes: scratch,
                maximum_allocations: self
                    .mechanisms
                    .scratch_allocation_count(operation.as_view())?,
                placement,
                host_control_bytes,
            });
        }
        let next_host = trace
            .host_workspace
            .checked_add(host.as_ref().map_or(0, |bound| bound.bytes))
            .ok_or_else(|| workspace_overflow("workspace host sum overflow"))?;
        let mut values: Vec<WorkspaceTensor> = self.metadata_vec(operation.outputs.len())?;
        self.reserve_metadata_vec(&mut trace.operations, 1)?;
        self.tracing_started.set(true);
        for (index, layout) in operation.outputs.iter().enumerate() {
            let population = bound
                .as_mut()
                .and_then(|bound| bound.take_output_population(index))
                .or_else(|| {
                    sources
                        .as_mut()
                        .and_then(|s| s.outputs.get_mut(index))
                        .and_then(Option::take)
                });
            let population = match population {
                Some(source) => Some(source),
                None if self.facts.is_none() => self
                    .mechanisms
                    .output_allocations(operation.as_view(), index)?
                    .cloned(),
                None => None,
            };
            let effect = bound.as_ref().and_then(|bound| bound.output(index));
            if population.is_some()
                && !matches!(
                    effect,
                    Some(WorkspaceOutputStorageView::Allocate(_))
                        | Some(WorkspaceOutputStorageView::AllocateOrAliasInputs { .. })
                )
            {
                return Err(WorkspaceMetadataError::Unqualified.into());
            }
            let storage = match effect {
                Some(WorkspaceOutputStorageView::AliasInput(input)) => {
                    inputs[input].storage.clone()
                }
                Some(WorkspaceOutputStorageView::AliasOutput(output)) => {
                    values[output].storage.clone()
                }
                effect if population.is_some() => {
                    let (bytes, candidates) = match effect {
                        Some(WorkspaceOutputStorageView::Allocate(bytes)) => (bytes, &[][..]),
                        Some(WorkspaceOutputStorageView::AllocateOrAliasInputs {
                            bytes,
                            inputs,
                        }) => (bytes, inputs),
                        _ => return Err(WorkspaceMetadataError::Unqualified.into()),
                    };
                    let mut aliases = self.metadata_vec(candidates.len())?;
                    aliases.extend(
                        candidates
                            .iter()
                            .map(|index| inputs[*index].storage.clone()),
                    );
                    self.output_population_storage(
                        population.expect("qualified population"),
                        bytes,
                        aliases,
                        &mut trace,
                    )?
                }
                effect => {
                    self.reserve_metadata_vec(&mut trace.allocations, 1)?;
                    let mut storage = self.try_new_storage(
                        match effect {
                            Some(WorkspaceOutputStorageView::Allocate(bytes))
                            | Some(WorkspaceOutputStorageView::AllocateOrAliasInputs {
                                bytes,
                                ..
                            }) => Some(bytes),
                            _ => None,
                        },
                        match effect {
                            Some(WorkspaceOutputStorageView::AllocateOrAliasInputs {
                                inputs: candidates,
                                ..
                            }) => {
                                let mut owners = self.metadata_vec(candidates.len())?;
                                owners.extend(
                                    candidates
                                        .iter()
                                        .map(|index| inputs[*index].storage.clone()),
                                );
                                owners
                            }
                            _ => Vec::new(),
                        },
                    )?;
                    Rc::get_mut(&mut storage)
                        .expect("unpublished workspace allocation")
                        .placement = self.copy_placement(
                        self.mechanisms.output_placement(operation.as_view(), index),
                    )?;
                    Rc::get_mut(&mut storage)
                        .expect("unpublished workspace allocation")
                        .host_control_bytes = self
                        .mechanisms
                        .allocation_host_control_bytes(operation.as_view(), index);
                    trace.allocations.push(storage.clone());
                    storage
                }
            };
            values.push(WorkspaceTensor {
                layout: layout.clone(),
                storage,
                context: self.identity.clone(),
                imported_existing: false,
            });
        }
        trace.scratch_overflow |= next_scratch.is_none();
        trace.scratch = next_scratch.unwrap_or(trace.scratch);
        trace.host_workspace = next_host;
        let missing_host = host.is_none();
        trace.host_staging_incomplete |= missing_host;
        if let Some(host) = host {
            self.insert_assumption(&mut trace.assumptions, host.assumptions)?;
        }
        if missing_host || missing_scratch_controls {
            let index = trace.operations.len();
            self.reserve_metadata_vec(&mut trace.missing_host, 1)?;
            trace.missing_host.push(index);
        }
        if let Some(bound) = bound {
            self.insert_assumption(&mut trace.assumptions, bound.into_assumptions())?;
        } else {
            let index = trace.operations.len();
            self.reserve_metadata_vec(&mut trace.missing, 1)?;
            trace.missing.push(index);
        }
        trace.operations.push(operation);
        Ok(values)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct WorkspaceOverflow(&'static str);

fn workspace_overflow(operation: &'static str) -> Error {
    Error::backend_retained_source(WorkspaceOverflow(operation))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "workspace/borrowed_tests.rs"]
mod borrowed_tests;

#[cfg(test)]
#[path = "workspace/placeholder_tests.rs"]
mod placeholder_tests;
