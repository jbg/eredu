//! Complete borrowed operation descriptions, without native values or authority.
use super::{
    LinearFormatSpec, WorkspaceBlockwisePolicy, WorkspaceBlockwiseStage, WorkspaceCollective,
    WorkspaceFloatingType, WorkspaceGroupedBank, WorkspaceGroupedPhase, WorkspaceLayout,
    WorkspaceLayoutView, WorkspaceOperation, WorkspaceOperationKind, WorkspaceSamplingOperation,
    WorkspaceStoredHostIdentity,
};

/// Borrows every ordinary operation option without cloning parameter or shape storage.
/// This is metadata; it grants no source custody or execution authority.
#[derive(Clone, Copy, Debug)]
pub enum WorkspaceOperationKindView<'a> {
    /// Exact selected independently addressable provider source.
    AddressableRegion(&'a super::WorkspaceAddressableRegion),
    /// Borrowed finite dynamic-region declaration.
    ExpertRegion(&'a super::WorkspaceExpertRegion),
    ExpertProviderWave(super::WorkspaceExpertProviderWave),
    ExpertInactiveWave(&'a super::WorkspaceExpertInactiveWave),
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
    /// Standalone effective checkpoint decoding with exact companion roles.
    ParameterDecode(crate::parameter_values::ParameterDecoding),
    /// Copy one retained immutable host value into independent device storage,
    /// preserving its exact floating type. The native source and completion
    /// account must be authenticated separately; this descriptor grants none.
    HostTransferFloating(WorkspaceFloatingType),
    /// Independent Host-store effect with an internal native completion root.
    HostStoreFloating(&'a WorkspaceStoredHostIdentity, WorkspaceFloatingType),
    /// Device materialization of an exact earlier opaque stored-Host value.
    HostLoadStoredFloating(&'a WorkspaceStoredHostIdentity, WorkspaceFloatingType),
    /// One fixed Rust F32 initialization buffer followed by the selected
    /// native host-input copy. No inputs and one F32 output; the scalar
    /// expression obeys Tensor::from_f32_fn's allocation-free contract.
    GeneratedF32Initialization,
    /// Sampling primitive selected by the shared runtime sampling policy.
    Sampling(&'a WorkspaceSamplingOperation),
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
    /// Exact normalized axis permutation borrowed from the retained trace.
    Transpose(&'a [usize]),
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
        starts: &'a [i32],
        /// Exclusive end on every axis.
        ends: &'a [i32],
        /// Strictly positive stride on every axis.
        strides: &'a [i32],
    },
    /// Unit-stride replacement of an exact rank-preserving rectangle. Native
    /// facts must include possible full destination copies and update casts;
    /// metadata never assumes that the destination can be donated in place.
    SliceUpdate {
        /// Nonnegative start on every axis; ends follow the update shape.
        starts: &'a [i32],
    },
    /// Exact positive-stride update without rank changes or update broadcasting.
    /// Native facts include a possible full destination copy and native casts.
    StaticSliceUpdate {
        /// Nonnegative start on every axis.
        starts: &'a [i32],
        /// Exclusive end on every axis.
        ends: &'a [i32],
        /// Strictly positive stride on every axis.
        strides: &'a [i32],
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
    Projection(&'a LinearFormatSpec),
    /// Observed block-FP8 preparation. Outputs are the actual compact U8
    /// activation and F32 block-scale roots; weight-scale decode stays live.
    ProjectionPrepare(&'a LinearFormatSpec),
    /// Observed block-FP8 projection after its callback. The last two inputs
    /// are exactly the compact activation/scale outputs of preparation.
    ProjectionFinish(&'a LinearFormatSpec),
    /// Fixed reconstruction of compact quantizer-produced U8 E4M3 values to
    /// F32. This is not a general arbitrary-stride conversion contract.
    BlockFp8ActivationDecode,
    /// Learned routing with the complete scoring and precision policy. Optional
    /// supplied IDs and intervention controls retain their distinct execution.
    GroupSelection {
        /// Exact construction and physical parameter topology.
        spec: &'a crate::TopKGroupSelectorSpec,
        /// Caller supplied the selected IDs instead of requesting top-k.
        supplied_indices: bool,
        /// Validated pre-dispatch control, including original-decision capture.
        control: Option<&'a crate::routing_intervention::GroupSelectionControl>,
    },
    /// Joint selected and always-on coefficient normalization.
    JointGroupSelection(&'a crate::JointGroupSelectionSpec),
    /// Selected grouped computation. The mechanism must bound every route
    /// distribution consistent with these shapes, including sorting and staging.
    Grouped {
        /// Retained architecture equation and exact parameter organization.
        bank: &'a WorkspaceGroupedBank,
        /// Whole equation or the two sides of an observed unit boundary.
        phase: WorkspaceGroupedPhase,
        /// Rank-local partial; replicated output bias is a separate result.
        partitions: Option<usize>,
    },
    /// FP32 residual preparation, learned mixing, Sinkhorn passes and collapse.
    HyperCollapse(&'a crate::HyperConnectionSpec, f32),
    /// Sublayer injection and residual mixing using retained coefficients.
    HyperExpand,
    /// Learned final stream coefficients, before an optional observer.
    HyperHeadCoefficients(&'a crate::HyperHeadSpec),
    /// Final weighted stream sum consuming the observed coefficients.
    HyperHeadSum,
    /// Row-parallel affine operation, including reduction and native bias order.
    RowParallelProjection(&'a LinearFormatSpec, usize),
    /// Embedding lookup with exact encoding and index validation/sentinel policy.
    Embedding(&'a LinearFormatSpec, crate::EmbeddingLookupPolicy),
    /// Global token validation, local row selection and zeroing contributions
    /// outside one rank's ownership. The following sum is traced separately.
    VocabularyParallelLookup {
        /// Exact local parameter encoding and companions.
        format: &'a LinearFormatSpec,
        /// Architecture-declared global and local vocabulary coordinates.
        range: &'a crate::VocabularyParallelRange,
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
    ConstructedNormalization(&'a crate::NormalizationConstructionSpec),
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
    MultiAxisRotary(crate::multimodal::MultiAxisRotarySpecRef<'a>),
    /// The same tensor operations with immutable caller-supplied host frequencies.
    /// Metadata is ordinary tracing storage; this carries no source-account credit.
    PreparedMultiAxisRotary(crate::multimodal::MultiAxisRotarySpecRef<'a>),
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
    /// Exact model-internal control source; native execution remains separately admitted.
    CommunicationControl(super::WorkspaceModelControl),
    /// Keeps actual existing values in the enclosing completion root union.
    ValueRetention,
    /// Boolean mask with prompt and cached-key geometry.
    CausalMask(crate::operation_geometry::CausalMaskGeometry),
    /// Eligibility of complete pooled source windows at exact query positions.
    PoolingMask(crate::operation_geometry::PoolingMaskGeometry),
    /// Convolution dimensions and normalized geometry.
    Convolution {
        /// One entry per spatial axis.
        stride: &'a [i32],
        /// Symmetric padding per spatial axis.
        padding: &'a [i32],
        /// Dilation per spatial axis.
        dilation: &'a [i32],
        /// Convolution groups.
        groups: i32,
        /// Output padding for transposed convolution; absent for forward.
        transposed: Option<&'a [i32]>,
    },
    /// Padding with the declared boundary rule.
    Pad(crate::PadMode),
    /// One rank's collective with exact peer geometry. Transport scratch must
    /// still be priced by the retained native mechanism.
    Collective(WorkspaceCollectiveView<'a>),
}
impl WorkspaceOperationKind {
    /// Borrows every allocation-relevant option without cloning dynamic storage.
    pub fn as_view(&self) -> WorkspaceOperationKindView<'_> {
        use WorkspaceOperationKindView as V;
        match self {
            Self::ParameterPlaceholder => V::ParameterPlaceholder,
            Self::ValueCompletion
            | Self::CommunicationDependencies
            | Self::CachePublicationCompletion
            | Self::CacheScanCompletion => V::ValueCompletion,
            Self::ValueRetention => V::ValueRetention,
            Self::CommunicationControl(control) => V::CommunicationControl(*control),
            Self::Initialize => V::Initialize,
            Self::InitializeFloating(dtype) => V::InitializeFloating(*dtype),
            Self::CastFloating(dtype) => V::CastFloating(*dtype),
            Self::HostTransferFloating(dtype) => V::HostTransferFloating(*dtype),
            Self::HostStoreFloating(identity, dtype) => V::HostStoreFloating(identity, *dtype),
            Self::HostLoadStoredFloating(identity, dtype) => {
                V::HostLoadStoredFloating(identity, *dtype)
            }
            Self::GeneratedF32Initialization => V::GeneratedF32Initialization,
            Self::AddressableRegion(region) => V::AddressableRegion(region),
            Self::ExpertRegion(region) => V::ExpertRegion(region),
            Self::ExpertProviderWave(wave) => V::ExpertProviderWave(*wave),
            Self::ExpertInactiveWave(wave) => V::ExpertInactiveWave(wave),
            Self::Sampling(a0) => V::Sampling(a0),
            Self::CandidateExtraction { vocabulary, count } => V::CandidateExtraction {
                vocabulary: *vocabulary,
                count: *count,
            },
            Self::Elementwise(a0) => V::Elementwise(*a0),
            Self::View(a0) => V::View(*a0),
            Self::Transpose(axes) => V::Transpose(axes),
            Self::Index { selected_axes } => V::Index {
                selected_axes: *selected_axes,
            },
            Self::StaticSlice {
                starts,
                ends,
                strides,
            } => V::StaticSlice {
                starts,
                ends,
                strides,
            },
            Self::SliceUpdate { starts } => V::SliceUpdate {
                starts: starts.as_slice(),
            },
            Self::StaticSliceUpdate {
                starts,
                ends,
                strides,
            } => V::StaticSliceUpdate {
                starts,
                ends,
                strides,
            },
            Self::Contiguous => V::Contiguous,
            Self::DeepCopy => V::DeepCopy,
            Self::Gather { axis } => V::Gather { axis: *axis },
            Self::IndexedRowAdd => V::IndexedRowAdd,
            Self::IndexedElementSelect => V::IndexedElementSelect,
            Self::IndexedElementUpdate => V::IndexedElementUpdate,
            Self::Concatenate => V::Concatenate,
            Self::Matmul => V::Matmul,
            Self::DenseLinear => V::DenseLinear,
            Self::Projection(a0) => V::Projection(a0),
            Self::ParameterDecode(decoding) => V::ParameterDecode(*decoding),
            Self::ProjectionPrepare(a0) => V::ProjectionPrepare(a0),
            Self::ProjectionFinish(a0) => V::ProjectionFinish(a0),
            Self::BlockFp8ActivationDecode => V::BlockFp8ActivationDecode,
            Self::GroupSelection {
                spec,
                supplied_indices,
                control,
            } => V::GroupSelection {
                spec: spec.as_ref(),
                supplied_indices: *supplied_indices,
                control: control.as_deref(),
            },
            Self::JointGroupSelection(a0) => V::JointGroupSelection(a0),
            Self::Grouped {
                bank,
                phase,
                partitions,
            } => V::Grouped {
                bank: bank.as_ref(),
                phase: *phase,
                partitions: *partitions,
            },
            Self::HyperCollapse(a0, a1) => V::HyperCollapse(a0.as_ref(), *a1),
            Self::HyperExpand => V::HyperExpand,
            Self::HyperHeadCoefficients(a0) => V::HyperHeadCoefficients(a0.as_ref()),
            Self::HyperHeadSum => V::HyperHeadSum,
            Self::RowParallelProjection(a0, a1) => V::RowParallelProjection(a0, *a1),
            Self::Embedding(a0, a1) => V::Embedding(a0, *a1),
            Self::VocabularyParallelLookup {
                format,
                range,
                policy,
            } => V::VocabularyParallelLookup {
                format: format,
                range: range,
                policy: *policy,
            },
            Self::Reduction(a0, a1, a2) => V::Reduction(*a0, *a1, *a2),
            Self::Normalization(a0, a1) => V::Normalization(*a0, *a1),
            Self::LayerNorm { weight, bias } => V::LayerNorm {
                weight: *weight,
                bias: *bias,
            },
            Self::ConstructedNormalization(a0) => V::ConstructedNormalization(a0),
            Self::GatedProduct(a0) => V::GatedProduct(*a0),
            Self::GatedDeltaScan => V::GatedDeltaScan,
            Self::SelectiveStateSpaceScan(a0, a1) => V::SelectiveStateSpaceScan(*a0, *a1),
            Self::Rotary(a0, a1) => V::Rotary(*a0, *a1),
            Self::RotaryFrequencies(a0, a1, a2) => V::RotaryFrequencies(*a0, *a1, *a2),
            Self::TensorRotary(a0, a1, a2, a3, a4) => V::TensorRotary(*a0, *a1, *a2, *a3, *a4),
            Self::MultiAxisRotary(a0) => V::MultiAxisRotary(a0.as_ref()),
            Self::PreparedMultiAxisRotary(a0) => V::PreparedMultiAxisRotary(a0.as_ref()),
            Self::MaskedOutputProjection {
                top_centroids,
                mask_margin,
            } => V::MaskedOutputProjection {
                top_centroids: *top_centroids,
                mask_margin: *mask_margin,
            },
            Self::IndexedAttention {
                scale,
                local_mask,
                pooled_mask,
                sinks,
            } => V::IndexedAttention {
                scale: *scale,
                local_mask: *local_mask,
                pooled_mask: *pooled_mask,
                sinks: *sinks,
            },
            Self::PooledAttention {
                scale,
                local_mask,
                pooled_mask,
                sinks,
            } => V::PooledAttention {
                scale: *scale,
                local_mask: *local_mask,
                pooled_mask: *pooled_mask,
                sinks: *sinks,
            },
            Self::PooledPositions {
                top_k,
                scale,
                head_scale,
                masked,
            } => V::PooledPositions {
                top_k: *top_k,
                scale: *scale,
                head_scale: *head_scale,
                masked: *masked,
            },
            Self::GatherPooledMask => V::GatherPooledMask,
            Self::RelativeAttention {
                query_offset,
                key_offset,
                window,
                log_scaling_floor,
                log_scaling_alpha,
            } => V::RelativeAttention {
                query_offset: *query_offset,
                key_offset: *key_offset,
                window: *window,
                log_scaling_floor: *log_scaling_floor,
                log_scaling_alpha: *log_scaling_alpha,
            },
            Self::Attention {
                causal,
                window,
                sinks,
                softcap,
                arithmetic,
            } => V::Attention {
                causal: *causal,
                window: *window,
                sinks: *sinks,
                softcap: *softcap,
                arithmetic: *arithmetic,
            },
            Self::BlockwiseAttention { policy, stage } => V::BlockwiseAttention {
                policy: *policy,
                stage: *stage,
            },
            Self::CausalMask(a0) => V::CausalMask(*a0),
            Self::PoolingMask(a0) => V::PoolingMask(*a0),
            Self::Convolution {
                stride,
                padding,
                dilation,
                groups,
                transposed,
            } => V::Convolution {
                stride: stride.as_slice(),
                padding: padding.as_slice(),
                dilation: dilation.as_slice(),
                groups: *groups,
                transposed: transposed.as_deref(),
            },
            Self::Pad(a0) => V::Pad(*a0),
            Self::Collective(a0) => V::Collective(a0.as_view()),
        }
    }
}

/// Borrowed per-rank collective geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceCollectiveView<'a> {
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
    /// Sum same-shaped contributions from all ranks.
    Sum {
        /// Participating ranks.
        partitions: usize,
        /// Local rank.
        rank: usize,
    },
    /// The exact output-publication Sum after its separately traced contribution.
    Broadcast {
        /// Retained selected publication group.
        group: eredu_core::CollectiveGroupId,
        /// Owner's local group index.
        root: usize,
        /// Selected group size.
        partitions: usize,
        /// Current local group index.
        rank: usize,
    },
    /// Native first-axis Gather of equally padded shards in the shared wrapper.
    GatherFirstAxis {
        /// Original axis padded before communication.
        axis: usize,
        /// Local rank.
        rank: usize,
        /// Original unpadded peer widths, in native rank order.
        peer_widths: &'a [usize],
    },
    /// Gather contiguous shards along the declared axis.
    Gather {
        /// Concatenation axis.
        axis: usize,
        /// Local rank.
        rank: usize,
        /// Actual per-peer widths, in rank order.
        peer_widths: &'a [usize],
    },
}
impl WorkspaceCollective {
    /// Borrows the exact collective geometry without copying the peer list.
    pub fn as_view(&self) -> WorkspaceCollectiveView<'_> {
        match self {
            Self::Boundary {
                route,
                ordinal,
                header_bytes,
            } => WorkspaceCollectiveView::Boundary {
                route: *route,
                ordinal: *ordinal,
                header_bytes: *header_bytes,
            },
            Self::Sum { partitions, rank } => WorkspaceCollectiveView::Sum {
                partitions: *partitions,
                rank: *rank,
            },
            Self::Broadcast {
                group,
                root,
                partitions,
                rank,
            } => WorkspaceCollectiveView::Broadcast {
                group: *group,
                root: *root,
                partitions: *partitions,
                rank: *rank,
            },
            Self::GatherFirstAxis {
                axis,
                rank,
                peer_widths,
            } => WorkspaceCollectiveView::GatherFirstAxis {
                axis: *axis,
                rank: *rank,
                peer_widths,
            },
            Self::Gather {
                axis,
                rank,
                peer_widths,
            } => WorkspaceCollectiveView::Gather {
                axis: *axis,
                rank: *rank,
                peer_widths,
            },
        }
    }
}

/// An ordered borrowed list of already-validated layouts.
#[derive(Clone, Copy, Debug)]
pub enum WorkspaceLayoutList<'a> {
    /// Existing ordinary layout storage.
    Owned(&'a [WorkspaceLayout]),
    /// Caller storage containing borrowed layout views.
    Views(&'a [WorkspaceLayoutView<'a>]),
}
impl<'a> WorkspaceLayoutList<'a> {
    /// Number of layouts in invocation order.
    pub const fn len(self) -> usize {
        match self {
            Self::Owned(v) => v.len(),
            Self::Views(v) => v.len(),
        }
    }
    /// Whether this list is empty.
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }
    /// Borrows one layout by its exact ordinal.
    pub fn get(self, index: usize) -> Option<WorkspaceLayoutView<'a>> {
        match self {
            Self::Owned(v) => v.get(index).map(WorkspaceLayout::as_view),
            Self::Views(v) => v.get(index).copied(),
        }
    }
    /// Borrows an ordered contiguous subset without constructing another list.
    pub fn slice(self, range: std::ops::Range<usize>) -> Option<Self> {
        match self {
            Self::Owned(v) => v.get(range).map(Self::Owned),
            Self::Views(v) => v.get(range).map(Self::Views),
        }
    }
    /// Iterates with exact size and without allocation.
    pub fn iter(self) -> WorkspaceLayoutIter<'a> {
        match self {
            Self::Owned(v) => WorkspaceLayoutIter::Owned(v.iter()),
            Self::Views(v) => WorkspaceLayoutIter::Views(v.iter()),
        }
    }
    /// Returns the first layout, if present.
    pub fn first(self) -> Option<WorkspaceLayoutView<'a>> {
        self.get(0)
    }
    /// Returns the last layout, if present.
    pub fn last(self) -> Option<WorkspaceLayoutView<'a>> {
        self.len().checked_sub(1).and_then(|i| self.get(i))
    }
    /// Borrows exactly the requested equation-local arity into stack storage.
    /// This imposes no general arity or rank limit.
    pub fn array<const N: usize>(self) -> Option<[WorkspaceLayoutView<'a>; N]> {
        if self.len() != N {
            return None;
        }
        Some(std::array::from_fn(|i| {
            self.get(i).expect("checked layout-list arity")
        }))
    }
}
impl PartialEq for WorkspaceLayoutList<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}
impl Eq for WorkspaceLayoutList<'_> {}

/// Allocation-free double-ended layout traversal.
#[derive(Clone, Debug)]
pub enum WorkspaceLayoutIter<'a> {
    /// Traversal over ordinary layouts.
    Owned(std::slice::Iter<'a, WorkspaceLayout>),
    /// Traversal over borrowed views.
    Views(std::slice::Iter<'a, WorkspaceLayoutView<'a>>),
}
impl<'a> Iterator for WorkspaceLayoutIter<'a> {
    type Item = WorkspaceLayoutView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(v) => v.next().map(WorkspaceLayout::as_view),
            Self::Views(v) => v.next().copied(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Owned(v) => v.size_hint(),
            Self::Views(v) => v.size_hint(),
        }
    }
}
impl DoubleEndedIterator for WorkspaceLayoutIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(v) => v.next_back().map(WorkspaceLayout::as_view),
            Self::Views(v) => v.next_back().copied(),
        }
    }
}
impl ExactSizeIterator for WorkspaceLayoutIter<'_> {}
impl std::iter::FusedIterator for WorkspaceLayoutIter<'_> {}

/// One complete operation borrowing its descriptions and ordered layouts.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceOperationView<'a> {
    /// Exact primitive and semantic options.
    pub kind: WorkspaceOperationKindView<'a>,
    /// Inputs in ordinary invocation order.
    pub inputs: WorkspaceLayoutList<'a>,
    /// Outputs in ordinary invocation order.
    pub outputs: WorkspaceLayoutList<'a>,
}
impl WorkspaceOperation {
    /// Borrows an operation without copying descriptions or shapes.
    pub fn as_view(&self) -> WorkspaceOperationView<'_> {
        WorkspaceOperationView {
            kind: self.kind.as_view(),
            inputs: WorkspaceLayoutList::Owned(&self.inputs),
            outputs: WorkspaceLayoutList::Owned(&self.outputs),
        }
    }
}

#[cfg(test)]
mod tests;
