//! Declarative physical checkpoint schemas.

#![allow(missing_docs)]

use std::collections::BTreeSet;

use eredu_gguf::GgmlType as GgufType;

use crate::{BlockFp8ScaleEncoding, LinearFormat, StoredDtype};

/// One named output segment in a fused projection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusedProjectionSegment {
    pub semantic: String,
    pub width: usize,
}

impl FusedProjectionSegment {
    pub fn new(semantic: impl Into<String>, width: usize) -> Result<Self, String> {
        let semantic = semantic.into();
        if semantic.trim().is_empty() || width == 0 {
            return Err("fused projection segments require a name and positive width".into());
        }
        Ok(Self { semantic, width })
    }
}

/// Family-neutral geometry for a projection whose output contains ordered
/// semantic segments sharing one input and partition domain.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusedSegmentedProjectionSchema {
    input_width: usize,
    segments: Vec<FusedProjectionSegment>,
    output_width: usize,
}

impl FusedSegmentedProjectionSchema {
    pub fn new(
        input_width: usize,
        segments: impl IntoIterator<Item = FusedProjectionSegment>,
    ) -> Result<Self, String> {
        if input_width == 0 {
            return Err("fused projection input width must be positive".into());
        }
        let segments = segments.into_iter().collect::<Vec<_>>();
        if segments.is_empty() {
            return Err("fused projection requires at least one segment".into());
        }
        let mut names = BTreeSet::new();
        let mut output_width = 0usize;
        for segment in &segments {
            if segment.semantic.trim().is_empty()
                || segment.width == 0
                || !names.insert(segment.semantic.clone())
            {
                return Err("fused projection segments must be positive and uniquely named".into());
            }
            output_width = output_width
                .checked_add(segment.width)
                .ok_or_else(|| "fused projection output width overflows".to_string())?;
        }
        Ok(Self {
            input_width,
            segments,
            output_width,
        })
    }

    pub const fn input_width(&self) -> usize {
        self.input_width
    }

    pub const fn output_width(&self) -> usize {
        self.output_width
    }

    pub fn matrix_shape(&self) -> Vec<usize> {
        vec![self.output_width, self.input_width]
    }

    pub fn bias_shape(&self) -> Vec<usize> {
        vec![self.output_width]
    }

    pub fn segment_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut start = 0;
        self.segments
            .iter()
            .map(|segment| {
                let range = start..start + segment.width;
                start = range.end;
                range
            })
            .collect()
    }
}

/// Physical axis convention for a depthwise causal-convolution kernel.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DepthwiseKernelAxes {
    /// `[channels, 1, kernel]`.
    ChannelsSingletonKernel,
    /// `[channels, kernel, 1]`.
    ChannelsKernelSingleton,
}

/// Explicit storage and execution geometry for a depthwise convolution.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DepthwiseConvolutionSchema {
    channels: usize,
    kernel: usize,
    storage_axes: DepthwiseKernelAxes,
    execution_axes: DepthwiseKernelAxes,
    bias: bool,
}

impl DepthwiseConvolutionSchema {
    pub fn new(channels: usize, kernel: usize, bias: bool) -> Result<Self, String> {
        Self::with_axes(
            channels,
            kernel,
            DepthwiseKernelAxes::ChannelsSingletonKernel,
            DepthwiseKernelAxes::ChannelsSingletonKernel,
            bias,
        )
    }

    pub fn with_axes(
        channels: usize,
        kernel: usize,
        storage_axes: DepthwiseKernelAxes,
        execution_axes: DepthwiseKernelAxes,
        bias: bool,
    ) -> Result<Self, String> {
        if channels == 0 || kernel == 0 {
            return Err("depthwise convolution channels and kernel must be positive".into());
        }
        Ok(Self {
            channels,
            kernel,
            storage_axes,
            execution_axes,
            bias,
        })
    }

    pub fn storage_shape(self) -> Vec<usize> {
        self.shape(self.storage_axes)
    }

    pub fn execution_shape(self) -> Vec<usize> {
        self.shape(self.execution_axes)
    }

    pub fn bias_shape(self) -> Option<Vec<usize>> {
        self.bias.then(|| vec![self.channels])
    }

    pub const fn element_count(self) -> usize {
        self.channels * self.kernel
    }

    fn shape(self, axes: DepthwiseKernelAxes) -> Vec<usize> {
        match axes {
            DepthwiseKernelAxes::ChannelsSingletonKernel => vec![self.channels, 1, self.kernel],
            DepthwiseKernelAxes::ChannelsKernelSingleton => vec![self.channels, self.kernel, 1],
        }
    }
}

/// Reusable head/group geometry for recurrent state-space parameter groups.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RecurrentParameterGroupSchema {
    pub heads: usize,
    pub groups: usize,
    pub head_width: usize,
    pub state_width: usize,
}

impl RecurrentParameterGroupSchema {
    pub fn new(
        heads: usize,
        groups: usize,
        head_width: usize,
        state_width: usize,
    ) -> Result<Self, String> {
        if heads == 0
            || groups == 0
            || head_width == 0
            || state_width == 0
            || !heads.is_multiple_of(groups)
        {
            return Err(
                "recurrent parameter groups require positive widths and whole groups".into(),
            );
        }
        Ok(Self {
            heads,
            groups,
            head_width,
            state_width,
        })
    }

    pub const fn per_head_shape(self) -> [usize; 1] {
        [self.heads]
    }

    pub const fn grouped_state_width(self) -> usize {
        self.groups * self.state_width
    }

    pub const fn recurrent_state_shape(self) -> [usize; 3] {
        [self.heads, self.head_width, self.state_width]
    }
}

/// Architecture-supplied physical names for a block-FP8 scale companion.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MatrixScaleNames {
    /// Canonical physical scale tensor name.
    pub key: String,
    /// Accepted alternative physical scale tensor names.
    pub aliases: Vec<String>,
}

/// Invalid matrix-plus-quantization-companion geometry.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum MatrixConstraintError {
    /// Matrix shape has no input dimension.
    #[error("quantized matrix {name:?} has scalar shape")]
    Scalar { name: String },
    /// Packing or companion geometry is incompatible.
    #[error("{detail}")]
    Invalid { detail: String },
    /// A block-FP8 matrix did not declare its architecture-owned scale name.
    #[error("block-FP8 matrix {name:?} requires an explicit scale companion name")]
    MissingBlockScaleName { name: String },
}

/// Builds a matrix constraint and every companion required by its complete
/// physical format.
///
/// This helper is family-neutral: callers provide canonical names, aliases,
/// logical shape, and the selected physical encoding.
pub fn matrix_for_linear_format(
    name: impl Into<String>,
    aliases: impl IntoIterator<Item = impl Into<String>>,
    shape: Vec<usize>,
    format: LinearFormat,
    scale_companion: Option<MatrixScaleNames>,
) -> Result<Vec<SafetensorsTensorConstraint>, MatrixConstraintError> {
    let name = name.into();
    let aliases = aliases.into_iter().map(Into::into).collect::<Vec<String>>();
    if format == LinearFormat::Dense {
        return Ok(vec![SafetensorsTensorConstraint::required(
            name,
            shape,
            StoredDtypeConstraint::Floating,
        )
        .with_aliases(aliases)]);
    }
    if let LinearFormat::E4M3BlockFp8(fp8) = format {
        fp8.validate()
            .map_err(|error| MatrixConstraintError::Invalid {
                detail: error.to_string(),
            })?;
        let scale = scale_companion
            .ok_or_else(|| MatrixConstraintError::MissingBlockScaleName { name: name.clone() })?;
        let row_axis = shape
            .len()
            .checked_sub(2)
            .ok_or_else(|| MatrixConstraintError::Scalar { name: name.clone() })?;
        let column_axis = shape.len() - 1;
        let rows = *shape
            .get(row_axis)
            .ok_or_else(|| MatrixConstraintError::Scalar { name: name.clone() })?;
        let columns = *shape
            .get(column_axis)
            .ok_or_else(|| MatrixConstraintError::Scalar { name: name.clone() })?;
        let block_rows =
            usize::try_from(fp8.block_rows).map_err(|_| MatrixConstraintError::Invalid {
                detail: format!("block-FP8 row geometry for {name:?} exceeds usize"),
            })?;
        let block_columns =
            usize::try_from(fp8.block_columns).map_err(|_| MatrixConstraintError::Invalid {
                detail: format!("block-FP8 column geometry for {name:?} exceeds usize"),
            })?;
        let mut scale_shape = shape.clone();
        scale_shape[row_axis] = rows.div_ceil(block_rows);
        scale_shape[column_axis] = columns.div_ceil(block_columns);
        let scale_dtype = match fp8.scale_encoding {
            BlockFp8ScaleEncoding::FloatingPoint => StoredDtypeConstraint::Floating,
            BlockFp8ScaleEncoding::Ue8m0 => StoredDtypeConstraint::Exact(StoredDtype::F8E8M0),
        };
        return Ok(vec![
            SafetensorsTensorConstraint::required(
                name,
                shape,
                StoredDtypeConstraint::Exact(StoredDtype::F8E4M3),
            )
            .with_aliases(aliases),
            SafetensorsTensorConstraint::required(scale.key, scale_shape, scale_dtype)
                .with_aliases(scale.aliases)
                .companion(),
        ]);
    }
    let quantization = format
        .weight_quantization()
        .expect("non-dense, non-FP8 linear formats use packed quantization");
    let input = *shape
        .last()
        .ok_or_else(|| MatrixConstraintError::Scalar { name: name.clone() })?;
    let bits =
        usize::try_from(quantization.bits()).map_err(|_| MatrixConstraintError::Invalid {
            detail: format!("quantization bit width for {name:?} exceeds usize"),
        })?;
    let group =
        usize::try_from(quantization.group_size()).map_err(|_| MatrixConstraintError::Invalid {
            detail: format!("quantization group size for {name:?} exceeds usize"),
        })?;
    let packed_bits = input
        .checked_mul(bits)
        .ok_or_else(|| MatrixConstraintError::Invalid {
            detail: format!("quantized matrix {name:?} packing geometry overflows"),
        })?;
    if group == 0
        || !input.is_multiple_of(group)
        || !input.is_multiple_of(32)
        || !packed_bits.is_multiple_of(32)
    {
        return Err(MatrixConstraintError::Invalid {
            detail: format!(
                "quantized matrix {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
            ),
        });
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    let prefix = name.strip_suffix(".weight").unwrap_or(&name).to_string();
    let weight_aliases = aliases
        .iter()
        .cloned()
        .chain(
            name.strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.inner.weight")),
        )
        .collect::<Vec<_>>();
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let companion_alias = |component: &str| {
        aliases
            .iter()
            .map(|alias| {
                let prefix = alias.strip_suffix(".weight").unwrap_or(alias);
                format!("{prefix}.{component}")
            })
            .collect::<Vec<_>>()
    };
    let mut constraints = vec![SafetensorsTensorConstraint::required(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )
    .with_aliases(weight_aliases)];
    let scale = scale_companion.unwrap_or_else(|| MatrixScaleNames {
        key: format!("{prefix}.scales"),
        aliases: companion_alias("scales"),
    });
    constraints.push(
        SafetensorsTensorConstraint::required(scale.key, companion.clone(), companion_dtype())
            .with_aliases(scale.aliases)
            .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(
                format!("{prefix}.biases"),
                companion,
                companion_dtype(),
            )
            .with_aliases(companion_alias("biases"))
            .companion(),
        );
    }
    Ok(constraints)
}

/// Whether a physical tensor must be present in the selected layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum TensorRequirement {
    Required,
    Optional,
}

/// How failures for a constraint are classified.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum TensorRole {
    Tensor,
    Companion,
}

/// Declarative SafeTensors storage constraint.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StoredDtypeConstraint {
    Exact(StoredDtype),
    OneOf(Vec<StoredDtype>),
    /// Repository-supported floating storage: F16, BF16, or F32.
    Floating,
}

impl StoredDtypeConstraint {
    pub fn accepts(&self, actual: &StoredDtype) -> bool {
        match self {
            Self::Exact(expected) => expected == actual,
            Self::OneOf(expected) => expected.contains(actual),
            Self::Floating => matches!(
                actual,
                StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32
            ),
        }
    }

    fn normalize(&mut self) {
        if let Self::OneOf(dtypes) = self {
            dtypes.sort_by_key(|dtype| format!("{dtype:?}"));
            dtypes.dedup();
        }
    }
}

/// One physical SafeTensors tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SafetensorsTensorConstraint {
    pub key: String,
    /// Alternative physical names for the same logical tensor.
    pub aliases: Vec<String>,
    pub shape: Vec<usize>,
    /// Additional accepted physical shapes with equivalent runtime semantics.
    pub alternate_shapes: Vec<Vec<usize>>,
    /// Accept any physical shape with this many elements. `shape` remains the
    /// canonical shape used by loading recipes.
    pub element_count: Option<usize>,
    pub dtype: StoredDtypeConstraint,
    pub requirement: TensorRequirement,
    pub role: TensorRole,
}

impl SafetensorsTensorConstraint {
    pub fn required(
        key: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        dtype: StoredDtypeConstraint,
    ) -> Self {
        Self {
            key: key.into(),
            aliases: Vec::new(),
            shape: shape.into(),
            alternate_shapes: Vec::new(),
            element_count: None,
            dtype,
            requirement: TensorRequirement::Required,
            role: TensorRole::Tensor,
        }
    }

    pub fn optional(mut self) -> Self {
        self.requirement = TensorRequirement::Optional;
        self
    }

    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_element_count(mut self, element_count: usize) -> Self {
        self.element_count = Some(element_count);
        self
    }

    pub fn with_alternate_shapes(
        mut self,
        shapes: impl IntoIterator<Item = impl Into<Vec<usize>>>,
    ) -> Self {
        self.alternate_shapes = shapes.into_iter().map(Into::into).collect();
        self
    }

    pub fn companion(mut self) -> Self {
        self.role = TensorRole::Companion;
        self
    }
}

/// Generic GGUF operation classes supported by runtime kernels.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum TensorOperation {
    Matrix,
    Vector,
    Dense,
    I32,
    MxFp4Matrix,
}

/// Declarative GGUF physical encoding constraint.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum GgufTypeConstraint {
    OperationClass(TensorOperation),
}

impl GgufTypeConstraint {
    pub fn accepts(&self, actual: GgufType) -> bool {
        match self {
            Self::OperationClass(operation) => gguf_encoding_supported(*operation, actual),
        }
    }

    fn normalize(&mut self) {}
}

/// Generic mapping from a numerical operation to accepted GGUF encodings.
pub fn gguf_encoding_supported(operation: TensorOperation, encoding: GgufType) -> bool {
    match operation {
        TensorOperation::Vector | TensorOperation::Dense => {
            matches!(encoding, GgufType::F32 | GgufType::F16 | GgufType::Bf16)
        }
        TensorOperation::I32 => encoding == GgufType::I32,
        TensorOperation::MxFp4Matrix => encoding == GgufType::MxFp4,
        TensorOperation::Matrix => !matches!(
            encoding,
            GgufType::I8
                | GgufType::I16
                | GgufType::I32
                | GgufType::I64
                | GgufType::F64
                | GgufType::RemovedIQ4NL4_4
                | GgufType::RemovedIQ4NL4_8
                | GgufType::RemovedIQ4NL8_8
                | GgufType::Unknown(_)
        ),
    }
}

/// One physical GGUF tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GgufTensorConstraint {
    pub key: String,
    /// Alternative physical names for the same logical tensor.
    pub aliases: Vec<String>,
    pub shape: Vec<usize>,
    /// Additional accepted physical shapes for encodings with equivalent
    /// runtime semantics (for example, flattened and singleton-axis kernels).
    pub alternate_shapes: Vec<Vec<usize>>,
    /// Accept any physical shape with this many elements. `shape` remains the
    /// canonical shape used by loading recipes.
    pub element_count: Option<usize>,
    pub encoding: GgufTypeConstraint,
    pub requirement: TensorRequirement,
    pub role: TensorRole,
}

impl GgufTensorConstraint {
    pub fn required(
        key: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        encoding: GgufTypeConstraint,
    ) -> Self {
        Self {
            key: key.into(),
            aliases: Vec::new(),
            shape: shape.into(),
            alternate_shapes: Vec::new(),
            element_count: None,
            encoding,
            requirement: TensorRequirement::Required,
            role: TensorRole::Tensor,
        }
    }

    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_alternate_shapes(
        mut self,
        shapes: impl IntoIterator<Item = impl Into<Vec<usize>>>,
    ) -> Self {
        self.alternate_shapes = shapes.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_element_count(mut self, element_count: usize) -> Self {
        self.element_count = Some(element_count);
        self
    }
}

/// One mutually exclusive physical layout.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LayoutVariant<T> {
    pub id: String,
    pub tensors: Vec<T>,
    pub discriminator_keys: Vec<String>,
}

/// A required or optional group of mutually exclusive layouts.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AlternativeLayoutGroup<T> {
    pub id: String,
    pub required: bool,
    pub variants: Vec<LayoutVariant<T>>,
}

/// Exact-catalog policy applied after selecting layouts.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CatalogPolicy {
    pub strict: bool,
    pub explicitly_allowed_keys: BTreeSet<String>,
    pub allowed_prefixes: Vec<String>,
    pub allowed_suffixes: Vec<String>,
}

impl CatalogPolicy {
    pub fn strict() -> Self {
        Self {
            strict: true,
            explicitly_allowed_keys: BTreeSet::new(),
            allowed_prefixes: Vec::new(),
            allowed_suffixes: Vec::new(),
        }
    }

    pub fn non_strict() -> Self {
        Self {
            strict: false,
            ..Self::strict()
        }
    }

    fn normalize(&mut self) {
        self.allowed_prefixes.sort();
        self.allowed_prefixes.dedup();
        self.allowed_suffixes.sort();
        self.allowed_suffixes.dedup();
    }
}

/// Invalid or ambiguous declarative plan.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CheckpointPlanError {
    #[error("checkpoint plan identity must not be empty")]
    EmptyIdentity,
    #[error("checkpoint plan contains an empty {kind} id")]
    EmptyId { kind: &'static str },
    #[error("checkpoint layout group {group:?} has no variants")]
    EmptyLayoutGroup { group: String },
    #[error("checkpoint layout variant {variant:?} has no tensors")]
    EmptyLayoutVariant { variant: String },
    #[error("checkpoint tensor key must not be empty")]
    EmptyTensorKey,
    #[error("checkpoint tensor {key:?} contains an empty physical alias")]
    EmptyTensorAlias { key: String },
    #[error("checkpoint tensor {key:?} has invalid shape {shape:?}")]
    InvalidShape { key: String, shape: Vec<usize> },
    #[error("checkpoint tensor {key:?} shape element count overflows")]
    ShapeOverflow { key: String },
    #[error("checkpoint tensor {key:?} has invalid required element count {element_count}")]
    InvalidElementCount { key: String, element_count: usize },
    #[error("checkpoint tensor {key:?} canonical shape contains {shape_elements} elements, but its required element count is {element_count}")]
    ElementCountMismatch {
        key: String,
        shape_elements: usize,
        element_count: usize,
    },
    #[error("checkpoint plan contains duplicate tensor key {key:?}")]
    DuplicateTensorKey { key: String },
    #[error("checkpoint layout variant {variant:?} has invalid discriminator {key:?}")]
    InvalidDiscriminator { variant: String, key: String },
    #[error("checkpoint plan contains duplicate {kind} id {id:?}")]
    DuplicateId { kind: &'static str, id: String },
    #[error("checkpoint catalog policy contains an empty explicitly allowed key")]
    EmptyAllowedKey,
    #[error("checkpoint catalog policy contains an empty allowed prefix")]
    EmptyAllowedPrefix,
    #[error("checkpoint catalog policy contains an empty allowed suffix")]
    EmptyAllowedSuffix,
    #[error("checkpoint tensor {key:?} has an empty encoding alternative set")]
    EmptyEncodingSet { key: String },
}

trait PhysicalConstraint {
    fn key(&self) -> &str;
    fn aliases(&self) -> &[String];
    fn aliases_mut(&mut self) -> &mut Vec<String>;
    fn shape(&self) -> &[usize];
    fn alternate_shapes(&self) -> &[Vec<usize>];
    fn element_count(&self) -> Option<usize>;
    fn normalize(&mut self);
    fn has_empty_encoding_set(&self) -> bool;
}

impl PhysicalConstraint for SafetensorsTensorConstraint {
    fn key(&self) -> &str {
        &self.key
    }
    fn aliases(&self) -> &[String] {
        &self.aliases
    }
    fn aliases_mut(&mut self) -> &mut Vec<String> {
        &mut self.aliases
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn alternate_shapes(&self) -> &[Vec<usize>] {
        &self.alternate_shapes
    }
    fn element_count(&self) -> Option<usize> {
        self.element_count
    }
    fn normalize(&mut self) {
        self.dtype.normalize();
        self.alternate_shapes.sort();
        self.alternate_shapes.dedup();
        self.alternate_shapes.retain(|shape| shape != &self.shape);
    }
    fn has_empty_encoding_set(&self) -> bool {
        matches!(&self.dtype, StoredDtypeConstraint::OneOf(dtypes) if dtypes.is_empty())
    }
}

impl PhysicalConstraint for GgufTensorConstraint {
    fn key(&self) -> &str {
        &self.key
    }
    fn aliases(&self) -> &[String] {
        &self.aliases
    }
    fn aliases_mut(&mut self) -> &mut Vec<String> {
        &mut self.aliases
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn alternate_shapes(&self) -> &[Vec<usize>] {
        &self.alternate_shapes
    }
    fn element_count(&self) -> Option<usize> {
        self.element_count
    }
    fn normalize(&mut self) {
        self.encoding.normalize();
        self.alternate_shapes.sort();
        self.alternate_shapes.dedup();
        self.alternate_shapes.retain(|shape| shape != &self.shape);
    }
    fn has_empty_encoding_set(&self) -> bool {
        false
    }
}

fn normalize_plan<T: PhysicalConstraint>(
    identity: &str,
    common: &mut [T],
    groups: &mut [AlternativeLayoutGroup<T>],
    policy: &mut CatalogPolicy,
) -> Result<(), CheckpointPlanError> {
    if identity.trim().is_empty() {
        return Err(CheckpointPlanError::EmptyIdentity);
    }
    let normalize_tensor = |tensor: &mut T| {
        tensor.normalize();
        if tensor.key().trim().is_empty() {
            return Err(CheckpointPlanError::EmptyTensorKey);
        }
        tensor.aliases_mut().sort();
        tensor.aliases_mut().dedup();
        if tensor.aliases().iter().any(|alias| alias.trim().is_empty()) {
            return Err(CheckpointPlanError::EmptyTensorAlias {
                key: tensor.key().into(),
            });
        }
        // Empty shapes are scalar tensors. A zero-sized dimension is invalid.
        let shapes = std::iter::once(tensor.shape()).chain(
            tensor
                .alternate_shapes()
                .iter()
                .map(|shape| shape.as_slice()),
        );
        let mut canonical_elements = None;
        for (index, shape) in shapes.enumerate() {
            if shape.contains(&0) {
                return Err(CheckpointPlanError::InvalidShape {
                    key: tensor.key().into(),
                    shape: shape.to_vec(),
                });
            }
            let elements = shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| CheckpointPlanError::ShapeOverflow {
                    key: tensor.key().into(),
                })?;
            if index == 0 {
                canonical_elements = Some(elements);
            }
        }
        if let Some(element_count) = tensor.element_count() {
            if element_count == 0 {
                return Err(CheckpointPlanError::InvalidElementCount {
                    key: tensor.key().into(),
                    element_count,
                });
            }
            if canonical_elements != Some(element_count) {
                return Err(CheckpointPlanError::ElementCountMismatch {
                    key: tensor.key().into(),
                    shape_elements: canonical_elements.expect("canonical shape was checked"),
                    element_count,
                });
            }
        }
        if tensor.has_empty_encoding_set() {
            return Err(CheckpointPlanError::EmptyEncodingSet {
                key: tensor.key().into(),
            });
        }
        Ok(())
    };
    let physical_keys = |tensor: &T| {
        std::iter::once(tensor.key().to_string())
            .chain(tensor.aliases().iter().cloned())
            .collect::<Vec<_>>()
    };
    let mut keys = BTreeSet::new();
    for tensor in common.iter_mut() {
        normalize_tensor(tensor)?;
        for physical_key in physical_keys(tensor) {
            if !keys.insert(physical_key.clone()) {
                return Err(CheckpointPlanError::DuplicateTensorKey { key: physical_key });
            }
        }
    }
    common.sort_by(|left, right| left.key().cmp(right.key()));

    let mut group_ids = BTreeSet::new();
    for group in groups.iter_mut() {
        if group.id.trim().is_empty() {
            return Err(CheckpointPlanError::EmptyId {
                kind: "layout group",
            });
        }
        if !group_ids.insert(group.id.clone()) {
            return Err(CheckpointPlanError::DuplicateId {
                kind: "layout group",
                id: group.id.clone(),
            });
        }
        if group.variants.is_empty() {
            return Err(CheckpointPlanError::EmptyLayoutGroup {
                group: group.id.clone(),
            });
        }
        let mut variant_ids = BTreeSet::new();
        let mut group_keys = BTreeSet::new();
        for variant in &mut group.variants {
            if variant.id.trim().is_empty() {
                return Err(CheckpointPlanError::EmptyId {
                    kind: "layout variant",
                });
            }
            if !variant_ids.insert(variant.id.clone()) {
                return Err(CheckpointPlanError::DuplicateId {
                    kind: "layout variant",
                    id: variant.id.clone(),
                });
            }
            if variant.tensors.is_empty() {
                return Err(CheckpointPlanError::EmptyLayoutVariant {
                    variant: variant.id.clone(),
                });
            }
            let mut variant_keys = keys.clone();
            for tensor in &mut variant.tensors {
                normalize_tensor(tensor)?;
                for physical_key in physical_keys(tensor) {
                    if !variant_keys.insert(physical_key.clone()) {
                        return Err(CheckpointPlanError::DuplicateTensorKey { key: physical_key });
                    }
                    group_keys.insert(physical_key);
                }
            }
            variant
                .tensors
                .sort_by(|left, right| left.key().cmp(right.key()));
            if variant.discriminator_keys.is_empty() {
                variant.discriminator_keys = variant
                    .tensors
                    .iter()
                    .map(|tensor| tensor.key().to_string())
                    .collect();
            }
            variant.discriminator_keys.sort();
            variant.discriminator_keys.dedup();
            let variant_keys = variant
                .tensors
                .iter()
                .map(|tensor| tensor.key())
                .collect::<BTreeSet<_>>();
            if let Some(key) = variant
                .discriminator_keys
                .iter()
                .find(|key| !variant_keys.contains(key.as_str()))
            {
                return Err(CheckpointPlanError::InvalidDiscriminator {
                    variant: variant.id.clone(),
                    key: key.clone(),
                });
            }
        }
        keys.extend(group_keys);
        group.variants.sort_by(|left, right| left.id.cmp(&right.id));
    }
    groups.sort_by(|left, right| left.id.cmp(&right.id));
    if policy
        .explicitly_allowed_keys
        .iter()
        .any(|key| key.trim().is_empty())
    {
        return Err(CheckpointPlanError::EmptyAllowedKey);
    }
    if policy
        .allowed_prefixes
        .iter()
        .any(|prefix| prefix.trim().is_empty())
    {
        return Err(CheckpointPlanError::EmptyAllowedPrefix);
    }
    if policy
        .allowed_suffixes
        .iter()
        .any(|suffix| suffix.trim().is_empty())
    {
        return Err(CheckpointPlanError::EmptyAllowedSuffix);
    }
    policy.normalize();
    Ok(())
}

macro_rules! checkpoint_plan {
    ($name:ident, $constraint:ty) => {
        #[derive(Debug, Clone, Eq, PartialEq)]
        pub struct $name {
            pub identity: String,
            pub common_tensors: Vec<$constraint>,
            pub layout_groups: Vec<AlternativeLayoutGroup<$constraint>>,
            pub catalog_policy: CatalogPolicy,
        }

        impl $name {
            pub fn new(
                identity: impl Into<String>,
                mut common_tensors: Vec<$constraint>,
                mut layout_groups: Vec<AlternativeLayoutGroup<$constraint>>,
                mut catalog_policy: CatalogPolicy,
            ) -> Result<Self, CheckpointPlanError> {
                let identity = identity.into();
                normalize_plan(
                    &identity,
                    &mut common_tensors,
                    &mut layout_groups,
                    &mut catalog_policy,
                )?;
                Ok(Self {
                    identity,
                    common_tensors,
                    layout_groups,
                    catalog_policy,
                })
            }
        }
    };
}

checkpoint_plan!(SafetensorsCheckpointPlan, SafetensorsTensorConstraint);
checkpoint_plan!(GgufCheckpointPlan, GgufTensorConstraint);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockFp8Format, BlockFp8ScaleEncoding};

    #[test]
    fn block_fp8_matrix_declares_exact_weight_scale_geometry_and_dtype() {
        let constraints = matrix_for_linear_format(
            "projection.weight",
            ["projection.alias.weight"],
            vec![8, 257, 129],
            LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0).unwrap(),
            ),
            Some(MatrixScaleNames {
                key: "projection.weight_scale_inv".into(),
                aliases: vec!["projection.alias.weight_scale_inv".into()],
            }),
        )
        .unwrap();
        assert_eq!(constraints.len(), 2);
        assert_eq!(constraints[0].shape, vec![8, 257, 129]);
        assert_eq!(
            constraints[0].dtype,
            StoredDtypeConstraint::Exact(StoredDtype::F8E4M3)
        );
        assert_eq!(constraints[1].shape, vec![8, 3, 2]);
        assert_eq!(
            constraints[1].dtype,
            StoredDtypeConstraint::Exact(StoredDtype::F8E8M0)
        );
        assert_eq!(constraints[1].role, TensorRole::Companion);
    }

    #[test]
    fn block_fp8_matrix_requires_architecture_supplied_scale_identity() {
        assert!(matches!(
            matrix_for_linear_format(
                "projection.weight",
                Vec::<String>::new(),
                vec![128, 128],
                LinearFormat::E4M3BlockFp8(
                    BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint,).unwrap(),
                ),
                None,
            ),
            Err(MatrixConstraintError::MissingBlockScaleName { .. })
        ));
    }

    #[test]
    fn construction_sorts_and_rejects_duplicate_or_invalid_shapes() {
        let tensor = |key: &str, shape| {
            SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
        };
        let plan = SafetensorsCheckpointPlan::new(
            "stable",
            vec![
                tensor("z", vec![2]),
                tensor("a", vec![1]),
                tensor("scalar", vec![]),
            ],
            Vec::new(),
            CatalogPolicy::strict(),
        )
        .unwrap();
        assert_eq!(
            plan.common_tensors
                .iter()
                .map(|tensor| tensor.key.as_str())
                .collect::<Vec<_>>(),
            ["a", "scalar", "z"]
        );
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "duplicate",
                vec![tensor("a", vec![1]), tensor("a", vec![1])],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::DuplicateTensorKey { .. })
        ));
        let aliased = tensor("logical", vec![1]).with_aliases(["physical"]);
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "duplicate alias",
                vec![aliased, tensor("physical", vec![1])],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::DuplicateTensorKey { key }) if key == "physical"
        ));
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "zero",
                vec![tensor("a", vec![0])],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::InvalidShape { .. })
        ));
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "overflow",
                vec![tensor("a", vec![usize::MAX, 2])],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::ShapeOverflow { .. })
        ));
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "invalid element count",
                vec![tensor("a", vec![2, 2]).with_element_count(3)],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::ElementCountMismatch { .. })
        ));
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "zero element count",
                vec![tensor("a", vec![2, 2]).with_element_count(0)],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::InvalidElementCount { .. })
        ));
        assert!(matches!(
            GgufCheckpointPlan::new(
                "invalid alternate",
                vec![GgufTensorConstraint::required(
                    "a",
                    vec![1],
                    GgufTypeConstraint::OperationClass(TensorOperation::Dense),
                )
                .with_alternate_shapes([vec![1, 0]])],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::InvalidShape { .. })
        ));
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "invalid SafeTensors alternate",
                vec![tensor("a", vec![1]).with_alternate_shapes([vec![1, 0]])],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::InvalidShape { .. })
        ));
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "empty encoding",
                vec![SafetensorsTensorConstraint::required(
                    "a",
                    vec![1],
                    StoredDtypeConstraint::OneOf(Vec::new()),
                )],
                Vec::new(),
                CatalogPolicy::strict(),
            ),
            Err(CheckpointPlanError::EmptyEncodingSet { .. })
        ));
        let mut empty_suffix = CatalogPolicy::strict();
        empty_suffix.allowed_suffixes.push(" ".into());
        assert!(matches!(
            SafetensorsCheckpointPlan::new(
                "empty allowed suffix",
                vec![tensor("a", vec![1])],
                Vec::new(),
                empty_suffix,
            ),
            Err(CheckpointPlanError::EmptyAllowedSuffix)
        ));

        let shared = tensor("shared", vec![1]);
        let sibling_shared = SafetensorsCheckpointPlan::new(
            "sibling shared key",
            Vec::new(),
            vec![AlternativeLayoutGroup {
                id: "layout".into(),
                required: true,
                variants: vec![
                    LayoutVariant {
                        id: "a".into(),
                        tensors: vec![tensor("a", vec![1]), shared.clone()],
                        discriminator_keys: vec!["a".into()],
                    },
                    LayoutVariant {
                        id: "b".into(),
                        tensors: vec![tensor("b", vec![1]), shared],
                        discriminator_keys: vec!["b".into()],
                    },
                ],
            }],
            CatalogPolicy::strict(),
        )
        .unwrap();
        assert_eq!(sibling_shared.layout_groups[0].variants.len(), 2);
    }

    #[test]
    fn hybrid_operator_schemas_freeze_segments_convolution_axes_and_recurrent_groups() {
        let projection = FusedSegmentedProjectionSchema::new(
            16,
            [
                FusedProjectionSegment::new("gate", 8).unwrap(),
                FusedProjectionSegment::new("state", 6).unwrap(),
                FusedProjectionSegment::new("time", 2).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(projection.matrix_shape(), [16, 16]);
        assert_eq!(projection.segment_ranges(), [0..8, 8..14, 14..16]);
        assert!(FusedSegmentedProjectionSchema::new(
            4,
            [
                FusedProjectionSegment::new("same", 2).unwrap(),
                FusedProjectionSegment::new("same", 2).unwrap(),
            ]
        )
        .is_err());

        let convolution = DepthwiseConvolutionSchema::with_axes(
            12,
            3,
            DepthwiseKernelAxes::ChannelsKernelSingleton,
            DepthwiseKernelAxes::ChannelsSingletonKernel,
            true,
        )
        .unwrap();
        assert_eq!(convolution.storage_shape(), [12, 3, 1]);
        assert_eq!(convolution.execution_shape(), [12, 1, 3]);
        assert_eq!(convolution.bias_shape(), Some(vec![12]));
        assert_eq!(convolution.element_count(), 36);

        let recurrent = RecurrentParameterGroupSchema::new(8, 2, 4, 3).unwrap();
        assert_eq!(recurrent.per_head_shape(), [8]);
        assert_eq!(recurrent.grouped_state_width(), 6);
        assert_eq!(recurrent.recurrent_state_shape(), [8, 4, 3]);
        assert!(RecurrentParameterGroupSchema::new(7, 2, 4, 3).is_err());
    }
}
