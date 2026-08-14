//! Declarative physical checkpoint schemas.

#![allow(dead_code)] // Some schema variants are reserved for staged architecture migrations.

use std::collections::BTreeSet;

use safemlx::ops::GgufType;

use super::store::StoredDtype;

/// Whether a physical tensor must be present in the selected layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum TensorRequirement {
    Required,
    Optional,
}

/// How failures for a constraint are classified.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum TensorRole {
    Tensor,
    Companion,
}

/// Declarative SafeTensors storage constraint.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum StoredDtypeConstraint {
    Exact(StoredDtype),
    OneOf(Vec<StoredDtype>),
    /// Repository-supported floating storage: F16, BF16, or F32.
    Floating,
}

impl StoredDtypeConstraint {
    pub(crate) fn accepts(&self, actual: &StoredDtype) -> bool {
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
pub(crate) struct SafetensorsTensorConstraint {
    pub(crate) key: String,
    /// Alternative physical names for the same logical tensor.
    pub(crate) aliases: Vec<String>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: StoredDtypeConstraint,
    pub(crate) requirement: TensorRequirement,
    pub(crate) role: TensorRole,
}

impl SafetensorsTensorConstraint {
    pub(crate) fn required(
        key: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        dtype: StoredDtypeConstraint,
    ) -> Self {
        Self {
            key: key.into(),
            aliases: Vec::new(),
            shape: shape.into(),
            dtype,
            requirement: TensorRequirement::Required,
            role: TensorRole::Tensor,
        }
    }

    pub(crate) fn optional(mut self) -> Self {
        self.requirement = TensorRequirement::Optional;
        self
    }

    pub(crate) fn with_aliases(
        mut self,
        aliases: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    pub(crate) fn companion(mut self) -> Self {
        self.role = TensorRole::Companion;
        self
    }
}

/// Generic GGUF operation classes supported by runtime kernels.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum TensorOperation {
    Matrix,
    Vector,
    Dense,
    I32,
    MxFp4Matrix,
}

/// Declarative GGUF physical encoding constraint.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum GgufTypeConstraint {
    Exact(GgufType),
    OneOf(Vec<GgufType>),
    OperationClass(TensorOperation),
}

impl GgufTypeConstraint {
    pub(crate) fn accepts(&self, actual: GgufType) -> bool {
        match self {
            Self::Exact(expected) => *expected == actual,
            Self::OneOf(expected) => expected.contains(&actual),
            Self::OperationClass(operation) => gguf_encoding_supported(*operation, actual),
        }
    }

    fn normalize(&mut self) {
        if let Self::OneOf(types) = self {
            types.sort_by_key(|encoding| encoding.code());
            types.dedup();
        }
    }
}

/// Generic mapping from a numerical operation to accepted GGUF encodings.
pub(crate) fn gguf_encoding_supported(operation: TensorOperation, encoding: GgufType) -> bool {
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
pub(crate) struct GgufTensorConstraint {
    pub(crate) key: String,
    /// Alternative physical names for the same logical tensor.
    pub(crate) aliases: Vec<String>,
    pub(crate) shape: Vec<usize>,
    /// Additional accepted physical shapes for encodings with equivalent
    /// runtime semantics (for example, flattened and singleton-axis kernels).
    pub(crate) alternate_shapes: Vec<Vec<usize>>,
    pub(crate) encoding: GgufTypeConstraint,
    pub(crate) requirement: TensorRequirement,
    pub(crate) role: TensorRole,
}

impl GgufTensorConstraint {
    pub(crate) fn required(
        key: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        encoding: GgufTypeConstraint,
    ) -> Self {
        Self {
            key: key.into(),
            aliases: Vec::new(),
            shape: shape.into(),
            alternate_shapes: Vec::new(),
            encoding,
            requirement: TensorRequirement::Required,
            role: TensorRole::Tensor,
        }
    }

    pub(crate) fn optional(mut self) -> Self {
        self.requirement = TensorRequirement::Optional;
        self
    }

    pub(crate) fn with_aliases(
        mut self,
        aliases: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    pub(crate) fn with_alternate_shapes(
        mut self,
        shapes: impl IntoIterator<Item = impl Into<Vec<usize>>>,
    ) -> Self {
        self.alternate_shapes = shapes.into_iter().map(Into::into).collect();
        self
    }

    pub(crate) fn companion(mut self) -> Self {
        self.role = TensorRole::Companion;
        self
    }
}

/// One mutually exclusive physical layout.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct LayoutVariant<T> {
    pub(crate) id: String,
    pub(crate) tensors: Vec<T>,
    pub(crate) discriminator_keys: Vec<String>,
}

/// A required or optional group of mutually exclusive layouts.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct AlternativeLayoutGroup<T> {
    pub(crate) id: String,
    pub(crate) required: bool,
    pub(crate) variants: Vec<LayoutVariant<T>>,
}

/// Exact-catalog policy applied after selecting layouts.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct CatalogPolicy {
    pub(crate) strict: bool,
    pub(crate) explicitly_allowed_keys: BTreeSet<String>,
    pub(crate) allowed_prefixes: Vec<String>,
}

impl CatalogPolicy {
    pub(crate) fn strict() -> Self {
        Self {
            strict: true,
            explicitly_allowed_keys: BTreeSet::new(),
            allowed_prefixes: Vec::new(),
        }
    }

    pub(crate) fn non_strict() -> Self {
        Self {
            strict: false,
            ..Self::strict()
        }
    }

    fn normalize(&mut self) {
        self.allowed_prefixes.sort();
        self.allowed_prefixes.dedup();
    }
}

/// Invalid or ambiguous declarative plan.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub(crate) enum CheckpointPlanError {
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
    #[error("checkpoint tensor {key:?} has an empty encoding alternative set")]
    EmptyEncodingSet { key: String },
}

trait PhysicalConstraint {
    fn key(&self) -> &str;
    fn aliases(&self) -> &[String];
    fn aliases_mut(&mut self) -> &mut Vec<String>;
    fn shape(&self) -> &[usize];
    fn alternate_shapes(&self) -> &[Vec<usize>];
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
        &[]
    }
    fn normalize(&mut self) {
        self.dtype.normalize();
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
    fn normalize(&mut self) {
        self.encoding.normalize();
        self.alternate_shapes.sort();
        self.alternate_shapes.dedup();
        self.alternate_shapes.retain(|shape| shape != &self.shape);
    }
    fn has_empty_encoding_set(&self) -> bool {
        matches!(&self.encoding, GgufTypeConstraint::OneOf(types) if types.is_empty())
    }
}

fn normalize_plan<T: PhysicalConstraint>(
    identity: &str,
    common: &mut Vec<T>,
    groups: &mut Vec<AlternativeLayoutGroup<T>>,
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
        for shape in shapes {
            if shape.contains(&0) {
                return Err(CheckpointPlanError::InvalidShape {
                    key: tensor.key().into(),
                    shape: shape.to_vec(),
                });
            }
            shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| CheckpointPlanError::ShapeOverflow {
                    key: tensor.key().into(),
                })?;
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
    policy.normalize();
    Ok(())
}

macro_rules! checkpoint_plan {
    ($name:ident, $constraint:ty) => {
        #[derive(Debug, Clone, Eq, PartialEq)]
        pub(crate) struct $name {
            pub(crate) identity: String,
            pub(crate) common_tensors: Vec<$constraint>,
            pub(crate) layout_groups: Vec<AlternativeLayoutGroup<$constraint>>,
            pub(crate) catalog_policy: CatalogPolicy,
        }

        impl $name {
            pub(crate) fn new(
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
            GgufCheckpointPlan::new(
                "invalid alternate",
                vec![GgufTensorConstraint::required(
                    "a",
                    vec![1],
                    GgufTypeConstraint::Exact(GgufType::F32),
                )
                .with_alternate_shapes([vec![1, 0]])],
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
}
