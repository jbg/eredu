//! Deterministic header-only evaluation of declarative checkpoint plans.

use crate::StoredDtype;

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorRequirement, TensorRole,
};
use eredu_gguf::{Checkpoint as GgufCheckpoint, GgmlType as GgufType};

/// Catalog metadata needed for header-only SafeTensors validation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CatalogTensorMetadata {
    /// Logical tensor shape.
    pub shape: Vec<usize>,
    /// Stored scalar encoding.
    pub stored_dtype: StoredDtype,
}

/// Backend-neutral SafeTensors catalog consumed by declarative validation.
pub trait SafetensorsCatalog {
    /// Returns all catalog keys in deterministic order.
    fn keys(&self) -> Vec<String>;
    /// Returns metadata without materializing tensor payloads.
    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String>;
}

impl SafetensorsCatalog for dyn crate::store::CheckpointSource + '_ {
    fn keys(&self) -> Vec<String> {
        self.source_keys()
    }

    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.source_metadata(key)
            .map(|metadata| CatalogTensorMetadata {
                shape: metadata.logical_shape,
                stored_dtype: metadata.stored_dtype,
            })
            .map_err(|error| error.to_string())
    }
}

/// Stable checkpoint validation categories used by inspection and strict load.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum CheckpointIssueKind {
    /// A required tensor is absent.
    MissingTensor,
    /// A tensor is not admitted by the selected contract.
    UnexpectedTensor,
    /// Mutually exclusive layouts conflict.
    ConflictingLayout,
    /// Tensor geometry does not match the contract.
    ShapeMismatch,
    /// Stored encoding is unsupported.
    UnsupportedEncoding,
    /// Atomic encoding companions are inconsistent.
    CompanionMismatch,
    /// Architecture geometry is invalid.
    InvalidGeometry,
    /// Validation could not be completed.
    ValidationUnavailable,
}

/// One structured checkpoint diagnostic.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CheckpointIssue {
    /// Stable diagnostic category.
    pub kind: CheckpointIssueKind,
    /// Human-readable detail.
    pub detail: String,
    /// Related tensor name, when applicable.
    pub tensor_name: Option<String>,
    /// Related format type code, when applicable.
    pub tensor_type_code: Option<u32>,
    /// Related metadata key, when applicable.
    pub metadata_key: Option<String>,
}

/// Result of exact, fail-closed checkpoint validation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CheckpointValidation {
    /// The catalog exactly satisfies the contract.
    Exact,
    /// The catalog violates the contract.
    Invalid(Vec<CheckpointIssue>),
    /// Validation was unavailable and loading must fail closed.
    Unverified(CheckpointIssue),
}

/// Neutral strict-load failure consumable by backend-specific error types.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("strict checkpoint validation failed")]
pub struct StrictLoadFailure {
    /// Missing required tensor names.
    pub missing: Vec<String>,
    /// Unexpected tensors and other validation details.
    pub unused: Vec<String>,
}

impl CheckpointValidation {
    /// Converts validation into a backend-neutral strict-load result.
    pub fn into_loader_result(self) -> Result<(), StrictLoadFailure> {
        match self {
            Self::Exact => Ok(()),
            Self::Unverified(issue) => Err(StrictLoadFailure {
                missing: Vec::new(),
                unused: vec![issue.detail],
            }),
            Self::Invalid(issues) => {
                let mut missing = issues
                    .iter()
                    .filter(|issue| issue.kind == CheckpointIssueKind::MissingTensor)
                    .filter_map(|issue| issue.tensor_name.clone())
                    .collect::<Vec<_>>();
                let mut unused = issues
                    .into_iter()
                    .filter(|issue| issue.kind != CheckpointIssueKind::MissingTensor)
                    .map(|issue| {
                        if issue.kind == CheckpointIssueKind::UnexpectedTensor {
                            issue.tensor_name.unwrap_or(issue.detail)
                        } else {
                            issue.detail
                        }
                    })
                    .collect::<Vec<_>>();
                missing.sort();
                missing.dedup();
                unused.sort();
                Err(StrictLoadFailure { missing, unused })
            }
        }
    }

    /// Applies a caller-selected strict catalog policy.
    pub fn with_strict_catalog(self, strict: bool) -> Self {
        if strict {
            return self;
        }
        match self {
            Self::Invalid(mut issues) => {
                issues.retain(|issue| issue.kind != CheckpointIssueKind::UnexpectedTensor);
                Self::from_issues(issues)
            }
            validation => validation,
        }
    }

    /// Builds exact or invalid validation from a diagnostic sequence.
    pub fn from_issues(issues: Vec<CheckpointIssue>) -> Self {
        if issues.is_empty() {
            Self::Exact
        } else {
            Self::Invalid(issues)
        }
    }
}

/// Builds a missing-tensor diagnostic.
pub fn missing(name: &str) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::MissingTensor,
        detail: format!("checkpoint is missing required tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

/// Builds a tensor shape diagnostic.
pub fn shape_mismatch(name: &str, expected: &[usize], actual: &[usize]) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::ShapeMismatch,
        detail: format!("tensor {name:?} expected shape {expected:?}, got {actual:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

/// The single physical layout selected from an architecture checkpoint plan.
///
/// Loaders consume this value through `ContractWeightStore`; they cannot read
/// aliases from an unselected layout or tensors that were not admitted by the
/// architecture contract.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResolvedCheckpointPlan {
    identity: String,
    source_keys: BTreeSet<String>,
    unclaimed_keys: BTreeSet<String>,
}

impl ResolvedCheckpointPlan {
    /// Returns the architecture checkpoint-plan identity.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Returns the exact selected physical source names.
    pub fn source_keys(&self) -> &BTreeSet<String> {
        &self.source_keys
    }

    /// Returns catalog names admitted but not claimed by a non-strict plan.
    pub fn unclaimed_keys(&self) -> &BTreeSet<String> {
        &self.unclaimed_keys
    }

    /// Creates a resolved plan for store contract tests.
    #[doc(hidden)]
    pub fn for_test(
        identity: impl Into<String>,
        source_keys: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            identity: identity.into(),
            source_keys: source_keys.into_iter().map(Into::into).collect(),
            unclaimed_keys: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PhysicalMetadata<E> {
    shape: Vec<usize>,
    encoding: E,
}

trait Constraint<E> {
    fn key(&self) -> &str;
    fn aliases(&self) -> &[String];
    fn shape(&self) -> &[usize];
    fn alternate_shapes(&self) -> &[Vec<usize>];
    fn element_count(&self) -> Option<usize>;
    fn requirement(&self) -> TensorRequirement;
    fn role(&self) -> TensorRole;
    fn accepts(&self, encoding: &E) -> bool;
    fn encoding_detail(&self) -> String;
    fn unsupported_detail(&self, identity: &str, actual: &E) -> String;
    fn type_code(&self, _actual: &E) -> Option<u32> {
        None
    }
}

impl Constraint<StoredDtype> for SafetensorsTensorConstraint {
    fn key(&self) -> &str {
        &self.key
    }
    fn aliases(&self) -> &[String] {
        &self.aliases
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
    fn requirement(&self) -> TensorRequirement {
        self.requirement
    }
    fn role(&self) -> TensorRole {
        self.role
    }
    fn accepts(&self, encoding: &StoredDtype) -> bool {
        self.dtype.accepts(encoding)
    }
    fn encoding_detail(&self) -> String {
        match &self.dtype {
            StoredDtypeConstraint::Exact(dtype) => format!("{dtype:?}"),
            StoredDtypeConstraint::OneOf(dtypes) => format!("one of {dtypes:?}"),
            StoredDtypeConstraint::Floating => "F16, BF16, or F32".into(),
        }
    }
    fn unsupported_detail(&self, _identity: &str, actual: &StoredDtype) -> String {
        format!(
            "tensor {:?} uses unsupported SafeTensors dtype {actual:?}; expected {}",
            self.key,
            self.encoding_detail()
        )
    }
}

impl Constraint<GgufType> for GgufTensorConstraint {
    fn key(&self) -> &str {
        &self.key
    }
    fn aliases(&self) -> &[String] {
        &self.aliases
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
    fn requirement(&self) -> TensorRequirement {
        self.requirement
    }
    fn role(&self) -> TensorRole {
        self.role
    }
    fn accepts(&self, encoding: &GgufType) -> bool {
        self.encoding.accepts(*encoding)
    }
    fn encoding_detail(&self) -> String {
        let GgufTypeConstraint::OperationClass(operation) = &self.encoding;
        format!("a {operation:?} operation")
    }
    fn unsupported_detail(&self, identity: &str, actual: &GgufType) -> String {
        format!(
            "GGUF tensor {:?} uses {actual:?} (type {}) for {}, which the {identity} loader does not support",
            self.key,
            actual.code(),
            self.encoding_detail()
        )
    }
    fn type_code(&self, actual: &GgufType) -> Option<u32> {
        Some(actual.code())
    }
}

/// Validates a SafeTensors store without materializing payloads.
pub fn validate_safetensors_plan(
    store: &(impl SafetensorsCatalog + ?Sized),
    plan: &SafetensorsCheckpointPlan,
) -> CheckpointValidation {
    let mut catalog = BTreeMap::new();
    let mut metadata_issues = Vec::new();
    for key in store.keys() {
        match store.metadata(&key) {
            Ok(CatalogTensorMetadata {
                shape,
                stored_dtype,
                ..
            }) => {
                catalog.insert(
                    key,
                    PhysicalMetadata {
                        shape,
                        encoding: stored_dtype,
                    },
                );
            }
            Err(error) => metadata_issues.push(metadata_failure(&key, error)),
        }
    }
    let mut issues = validate_catalog(
        &catalog,
        &plan.identity,
        &plan.common_tensors,
        &plan.layout_groups,
        &plan.catalog_policy,
    );
    metadata_issues.append(&mut issues);
    CheckpointValidation::from_issues(metadata_issues)
}

/// Resolves and validates the one SafeTensors layout that loading may consume.
pub fn resolve_safetensors_plan(
    store: &(impl SafetensorsCatalog + ?Sized),
    plan: &SafetensorsCheckpointPlan,
) -> Result<ResolvedCheckpointPlan, CheckpointValidation> {
    let mut catalog = BTreeMap::new();
    let mut metadata_issues = Vec::new();
    for key in store.keys() {
        match store.metadata(&key) {
            Ok(CatalogTensorMetadata {
                shape,
                stored_dtype,
                ..
            }) => {
                catalog.insert(
                    key,
                    PhysicalMetadata {
                        shape,
                        encoding: stored_dtype,
                    },
                );
            }
            Err(error) => metadata_issues.push(metadata_failure(&key, error)),
        }
    }
    resolve_catalog(
        &catalog,
        &plan.identity,
        &plan.common_tensors,
        &plan.layout_groups,
        &plan.catalog_policy,
        metadata_issues,
    )
}

/// Validates a GGUF catalog without decoding tensor payloads.
pub fn validate_gguf_plan(
    checkpoint: &GgufCheckpoint,
    plan: &GgufCheckpointPlan,
) -> CheckpointValidation {
    let catalog = gguf_catalog(checkpoint);
    CheckpointValidation::from_issues(validate_catalog(
        &catalog,
        &plan.identity,
        &plan.common_tensors,
        &plan.layout_groups,
        &plan.catalog_policy,
    ))
}

/// Resolves and validates the one GGUF layout that loading may consume.
pub fn resolve_gguf_plan(
    checkpoint: &GgufCheckpoint,
    plan: &GgufCheckpointPlan,
) -> Result<ResolvedCheckpointPlan, CheckpointValidation> {
    resolve_catalog(
        &gguf_catalog(checkpoint),
        &plan.identity,
        &plan.common_tensors,
        &plan.layout_groups,
        &plan.catalog_policy,
        Vec::new(),
    )
}

fn gguf_catalog(checkpoint: &GgufCheckpoint) -> BTreeMap<String, PhysicalMetadata<GgufType>> {
    checkpoint
        .tensors()
        .map(|tensor| {
            let descriptor = tensor.descriptor();
            (
                descriptor.name.clone(),
                PhysicalMetadata {
                    shape: descriptor
                        .row_major_shape()
                        .into_iter()
                        .map(|dimension| usize::try_from(dimension).unwrap_or(usize::MAX))
                        .collect(),
                    encoding: descriptor.ggml_type,
                },
            )
        })
        .collect()
}

fn resolve_catalog<E, T>(
    catalog: &BTreeMap<String, PhysicalMetadata<E>>,
    identity: &str,
    common: &[T],
    groups: &[AlternativeLayoutGroup<T>],
    policy: &CatalogPolicy,
    mut issues: Vec<CheckpointIssue>,
) -> Result<ResolvedCheckpointPlan, CheckpointValidation>
where
    E: std::fmt::Debug,
    T: Constraint<E>,
{
    issues.extend(validate_catalog(catalog, identity, common, groups, policy));
    if !issues.is_empty() {
        return Err(CheckpointValidation::from_issues(issues));
    }

    let mut source_keys = BTreeSet::new();
    let mut select_constraint = |constraint: &T| {
        if let Some(key) = std::iter::once(constraint.key())
            .chain(constraint.aliases().iter().map(String::as_str))
            .find(|key| catalog.contains_key(*key))
        {
            source_keys.insert(key.to_string());
        }
    };
    for constraint in common {
        select_constraint(constraint);
    }

    for group in groups {
        if let Some(variant) = group.variants.iter().find(|variant| {
            variant
                .discriminator_keys
                .iter()
                .all(|key| discriminator_present(catalog, variant, key))
        }) {
            for constraint in &variant.tensors {
                select_constraint(constraint);
            }
        }
    }

    let unclaimed_keys = if policy.strict {
        BTreeSet::new()
    } else {
        catalog
            .keys()
            .filter(|key| !source_keys.contains(*key))
            .filter(|key| !policy.explicitly_allowed_keys.contains(*key))
            .filter(|key| {
                !policy
                    .allowed_prefixes
                    .iter()
                    .any(|prefix| key.starts_with(prefix))
            })
            .cloned()
            .collect()
    };

    Ok(ResolvedCheckpointPlan {
        identity: identity.into(),
        source_keys,
        unclaimed_keys,
    })
}

/// Validates that architecture-supplied tensor pairs use identical encodings.
pub fn validate_matching_gguf_encodings(
    checkpoint: &GgufCheckpoint,
    pairs: impl IntoIterator<Item = (String, String)>,
    label: &str,
) -> Vec<CheckpointIssue> {
    validate_gguf_encoding_pairs(checkpoint, pairs, label, false)
}

/// Validates paired encodings while permitting different dense floating
/// storage types, which share the same dense runtime operation.
pub fn validate_dense_or_matching_gguf_encodings(
    checkpoint: &GgufCheckpoint,
    pairs: impl IntoIterator<Item = (String, String)>,
    label: &str,
) -> Vec<CheckpointIssue> {
    validate_gguf_encoding_pairs(checkpoint, pairs, label, true)
}

fn validate_gguf_encoding_pairs(
    checkpoint: &GgufCheckpoint,
    pairs: impl IntoIterator<Item = (String, String)>,
    label: &str,
    allow_mixed_dense: bool,
) -> Vec<CheckpointIssue> {
    let catalog = checkpoint
        .tensors()
        .map(|tensor| (tensor.descriptor().name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut issues = Vec::new();
    for (gate_name, up_name) in pairs {
        let (Some(gate), Some(up)) = (
            catalog.get(gate_name.as_str()),
            catalog.get(up_name.as_str()),
        ) else {
            continue;
        };
        let gate_type = gate.descriptor().ggml_type;
        let up_type = up.descriptor().ggml_type;
        let dense = |encoding| matches!(encoding, GgufType::F32 | GgufType::F16 | GgufType::Bf16);
        let compatible = (allow_mixed_dense && dense(gate_type) && dense(up_type))
            || (gate_type == up_type
                && gate.affine() == up.affine()
                && gate.is_mxfp4() == up.is_mxfp4());
        if !compatible {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::CompanionMismatch,
                detail: format!(
                    "{label} paired expert tensors {gate_name:?} and {up_name:?} use incompatible encodings {:?} and {:?}",
                    gate_type, up_type
                ),
                tensor_name: Some(gate_name),
                tensor_type_code: Some(gate_type.code()),
                metadata_key: None,
            });
        }
    }
    issues
}

fn validate_catalog<E, T>(
    catalog: &BTreeMap<String, PhysicalMetadata<E>>,
    identity: &str,
    common: &[T],
    groups: &[AlternativeLayoutGroup<T>],
    policy: &CatalogPolicy,
) -> Vec<CheckpointIssue>
where
    E: std::fmt::Debug,
    T: Constraint<E>,
{
    let mut issues = Vec::new();
    let mut accounted = BTreeSet::new();
    for constraint in common {
        account_constraint(&mut accounted, constraint);
        validate_constraint(catalog, identity, constraint, &mut issues);
    }

    for group in groups {
        let mut present = Vec::new();
        let mut partial = Vec::new();
        for variant in &group.variants {
            let count = variant
                .discriminator_keys
                .iter()
                .filter(|key| discriminator_present(catalog, variant, key))
                .count();
            if count == variant.discriminator_keys.len() {
                present.push(variant);
            } else if count != 0 {
                partial.push((variant, count));
            }
        }

        for (variant, count) in &partial {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::ConflictingLayout,
                detail: format!(
                    "checkpoint plan {:?} layout group {:?} has partially present variant {:?}: {count}/{} discriminator tensors are present",
                    identity,
                    group.id,
                    variant.id,
                    variant.discriminator_keys.len()
                ),
                tensor_name: variant
                    .discriminator_keys
                    .iter()
                    .find(|key| !discriminator_present(catalog, variant, key))
                    .cloned(),
                tensor_type_code: None,
                metadata_key: None,
            });
        }

        if !present.is_empty() && !partial.is_empty() {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::ConflictingLayout,
                detail: format!(
                    "checkpoint plan {:?} layout group {:?} mixes present variants {:?} with partially present variants {:?}",
                    identity,
                    group.id,
                    present
                        .iter()
                        .map(|variant| variant.id.as_str())
                        .collect::<Vec<_>>(),
                    partial
                        .iter()
                        .map(|(variant, _)| variant.id.as_str())
                        .collect::<Vec<_>>()
                ),
                tensor_name: present
                    .first()
                    .and_then(|variant| variant.discriminator_keys.first())
                    .cloned(),
                tensor_type_code: None,
                metadata_key: None,
            });
        }

        if present.len() > 1 {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::ConflictingLayout,
                detail: format!(
                    "checkpoint plan {:?} layout group {:?} has conflicting variants {:?}",
                    identity,
                    group.id,
                    present
                        .iter()
                        .map(|variant| variant.id.as_str())
                        .collect::<Vec<_>>()
                ),
                tensor_name: present
                    .get(1)
                    .and_then(|variant| variant.discriminator_keys.first())
                    .cloned(),
                tensor_type_code: None,
                metadata_key: None,
            });
        } else if present.is_empty() && partial.is_empty() && group.required {
            let missing_key = group
                .variants
                .first()
                .and_then(|variant| variant.discriminator_keys.first())
                .cloned();
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::MissingTensor,
                detail: format!(
                    "checkpoint plan {:?} has no matching variant for required layout group {:?}",
                    identity, group.id
                ),
                tensor_name: missing_key,
                tensor_type_code: None,
                metadata_key: None,
            });
        }

        for variant in present {
            for constraint in &variant.tensors {
                account_constraint(&mut accounted, constraint);
                validate_constraint(catalog, identity, constraint, &mut issues);
            }
        }
        for (variant, _) in partial {
            for constraint in &variant.tensors {
                if constraint_present(catalog, constraint) {
                    account_constraint(&mut accounted, constraint);
                }
                validate_constraint(catalog, identity, constraint, &mut issues);
            }
        }
    }

    if policy.strict {
        for key in catalog.keys() {
            if accounted.contains(key)
                || policy.explicitly_allowed_keys.contains(key)
                || policy
                    .allowed_prefixes
                    .iter()
                    .any(|prefix| key.starts_with(prefix))
            {
                continue;
            }
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::UnexpectedTensor,
                detail: format!("{identity} catalog contains unexpected tensor {key:?}"),
                tensor_name: Some(key.clone()),
                tensor_type_code: None,
                metadata_key: None,
            });
        }
    }
    issues
}

fn constraint_present<E, T: Constraint<E>>(
    catalog: &BTreeMap<String, PhysicalMetadata<E>>,
    constraint: &T,
) -> bool {
    std::iter::once(constraint.key())
        .chain(constraint.aliases().iter().map(String::as_str))
        .any(|key| catalog.contains_key(key))
}

fn discriminator_present<E, T: Constraint<E>>(
    catalog: &BTreeMap<String, PhysicalMetadata<E>>,
    variant: &crate::schema::LayoutVariant<T>,
    key: &str,
) -> bool {
    variant
        .tensors
        .iter()
        .find(|constraint| constraint.key() == key)
        .map_or_else(
            || catalog.contains_key(key),
            |constraint| constraint_present(catalog, constraint),
        )
}

fn account_constraint<E, T: Constraint<E>>(accounted: &mut BTreeSet<String>, constraint: &T) {
    accounted.insert(constraint.key().to_string());
    accounted.extend(constraint.aliases().iter().cloned());
}

fn validate_constraint<E: std::fmt::Debug, T: Constraint<E>>(
    catalog: &BTreeMap<String, PhysicalMetadata<E>>,
    identity: &str,
    constraint: &T,
    issues: &mut Vec<CheckpointIssue>,
) {
    let present = std::iter::once(constraint.key())
        .chain(constraint.aliases().iter().map(String::as_str))
        .filter_map(|key| catalog.get(key).map(|metadata| (key, metadata)))
        .collect::<Vec<_>>();
    if present.len() > 1 {
        issues.push(CheckpointIssue {
            kind: CheckpointIssueKind::ConflictingLayout,
            detail: format!(
                "checkpoint plan {identity:?} contains multiple physical aliases for logical tensor {:?}: {:?}",
                constraint.key(),
                present.iter().map(|(key, _)| *key).collect::<Vec<_>>()
            ),
            tensor_name: present.get(1).map(|(key, _)| (*key).to_string()),
            tensor_type_code: None,
            metadata_key: None,
        });
        return;
    }
    let Some((actual_key, actual)) = present.first().copied() else {
        if constraint.requirement() == TensorRequirement::Required {
            if constraint.role() == TensorRole::Companion {
                issues.push(companion_issue(
                    constraint.key(),
                    format!(
                        "checkpoint is missing required companion tensor {:?}",
                        constraint.key()
                    ),
                ));
            } else {
                issues.push(missing(constraint.key()));
            }
        }
        return;
    };

    if constraint.role() == TensorRole::Companion
        && (!accepts_shape(constraint, &actual.shape) || !constraint.accepts(&actual.encoding))
    {
        issues.push(companion_issue(
            actual_key,
            format!(
                "companion tensor {:?} expected shape {:?} and {}, got {:?} {:?}",
                actual_key,
                constraint.shape(),
                constraint.encoding_detail(),
                actual.shape,
                actual.encoding
            ),
        ));
        return;
    }

    if !accepts_shape(constraint, &actual.shape) {
        if let Some(element_count) = constraint.element_count() {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::ShapeMismatch,
                detail: format!(
                    "tensor {actual_key:?} must contain {element_count} elements for the loader transform, got {:?}",
                    actual.shape
                ),
                tensor_name: Some(actual_key.into()),
                tensor_type_code: constraint.type_code(&actual.encoding),
                metadata_key: None,
            });
        } else if constraint.alternate_shapes().is_empty() {
            issues.push(shape_mismatch(
                actual_key,
                constraint.shape(),
                &actual.shape,
            ));
        } else {
            let expected = std::iter::once(constraint.shape())
                .chain(
                    constraint
                        .alternate_shapes()
                        .iter()
                        .map(|shape| shape.as_slice()),
                )
                .collect::<Vec<_>>();
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::ShapeMismatch,
                detail: format!(
                    "tensor {actual_key:?} expected one of shapes {expected:?}, got {:?}",
                    actual.shape
                ),
                tensor_name: Some(actual_key.into()),
                tensor_type_code: None,
                metadata_key: None,
            });
        }
    }
    if !constraint.accepts(&actual.encoding) {
        issues.push(CheckpointIssue {
            kind: CheckpointIssueKind::UnsupportedEncoding,
            detail: constraint.unsupported_detail(identity, &actual.encoding),
            tensor_name: Some(actual_key.into()),
            tensor_type_code: constraint.type_code(&actual.encoding),
            metadata_key: None,
        });
    }
}

fn accepts_shape<E, T: Constraint<E>>(constraint: &T, actual: &[usize]) -> bool {
    if let Some(element_count) = constraint.element_count() {
        return actual
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            == Some(element_count);
    }
    actual == constraint.shape()
        || constraint
            .alternate_shapes()
            .iter()
            .any(|shape| shape == actual)
}

fn companion_issue(name: &str, detail: String) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::CompanionMismatch,
        detail,
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: Some("quantization".into()),
    }
}

fn metadata_failure(name: &str, error: String) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::ConflictingLayout,
        detail: format!("could not validate tensor {name:?}: {error}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{LayoutVariant, TensorOperation};

    fn safe(
        key: &str,
        shape: &[usize],
        dtype: StoredDtype,
    ) -> (String, PhysicalMetadata<StoredDtype>) {
        (
            key.into(),
            PhysicalMetadata {
                shape: shape.to_vec(),
                encoding: dtype,
            },
        )
    }

    fn safe_plan(
        common: Vec<SafetensorsTensorConstraint>,
        groups: Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
        policy: CatalogPolicy,
    ) -> SafetensorsCheckpointPlan {
        SafetensorsCheckpointPlan::new("test", common, groups, policy).unwrap()
    }

    #[test]
    fn required_optional_unexpected_and_catalog_exclusions_are_generic() {
        let plan = safe_plan(
            vec![
                SafetensorsTensorConstraint::required(
                    "required",
                    vec![2],
                    StoredDtypeConstraint::Floating,
                ),
                SafetensorsTensorConstraint::required(
                    "optional",
                    vec![1],
                    StoredDtypeConstraint::Floating,
                )
                .optional(),
            ],
            Vec::new(),
            CatalogPolicy {
                strict: true,
                explicitly_allowed_keys: BTreeSet::from(["allowed".into()]),
                allowed_prefixes: vec!["cache.".into()],
            },
        );
        let catalog = BTreeMap::from([
            safe("required", &[2], StoredDtype::F16),
            safe("allowed", &[1], StoredDtype::I32),
            safe("cache.value", &[1], StoredDtype::I32),
            safe("unexpected", &[1], StoredDtype::F32),
        ]);
        let issues = validate_catalog(
            &catalog,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        );
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].kind, CheckpointIssueKind::UnexpectedTensor);

        let mut non_strict = plan.clone();
        non_strict.catalog_policy.strict = false;
        assert!(validate_catalog(
            &catalog,
            &non_strict.identity,
            &non_strict.common_tensors,
            &non_strict.layout_groups,
            &non_strict.catalog_policy,
        )
        .is_empty());
    }

    #[test]
    fn exact_floating_and_encoded_fp8_constraints_keep_storage_distinct() {
        let constraints = vec![
            SafetensorsTensorConstraint::required(
                "exact",
                vec![1],
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            ),
            SafetensorsTensorConstraint::required(
                "weight",
                vec![2, 2],
                StoredDtypeConstraint::OneOf(vec![StoredDtype::F8E4M3, StoredDtype::U8]),
            ),
            SafetensorsTensorConstraint::required(
                "scale",
                vec![1, 1],
                StoredDtypeConstraint::Floating,
            )
            .companion(),
        ];
        let plan = safe_plan(constraints, Vec::new(), CatalogPolicy::strict());
        for floating in [StoredDtype::F16, StoredDtype::BF16, StoredDtype::F32] {
            let catalog = BTreeMap::from([
                safe("exact", &[1], StoredDtype::U8),
                safe("weight", &[2, 2], StoredDtype::F8E4M3),
                safe("scale", &[1, 1], floating),
            ]);
            assert!(validate_catalog(
                &catalog,
                &plan.identity,
                &plan.common_tensors,
                &plan.layout_groups,
                &plan.catalog_policy,
            )
            .is_empty());
        }
        let catalog = BTreeMap::from([
            safe("exact", &[1], StoredDtype::U8),
            safe("weight", &[2, 2], StoredDtype::U8),
            safe("scale", &[1, 1], StoredDtype::I32),
        ]);
        let issues = validate_catalog(
            &catalog,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        );
        assert_eq!(issues[0].kind, CheckpointIssueKind::CompanionMismatch);
    }

    #[test]
    fn physical_aliases_are_selected_once_and_accounted_by_strict_catalogs() {
        let plan = safe_plan(
            vec![SafetensorsTensorConstraint::required(
                "canonical",
                vec![2],
                StoredDtypeConstraint::Floating,
            )
            .with_aliases(["released"])],
            Vec::new(),
            CatalogPolicy::strict(),
        );
        let released = BTreeMap::from([safe("released", &[2], StoredDtype::BF16)]);
        assert!(validate_catalog(
            &released,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        )
        .is_empty());

        let conflicting = BTreeMap::from([
            safe("canonical", &[2], StoredDtype::F16),
            safe("released", &[2], StoredDtype::BF16),
        ]);
        let issues = validate_catalog(
            &conflicting,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        );
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].kind, CheckpointIssueKind::ConflictingLayout);
        assert_eq!(issues[0].tensor_name.as_deref(), Some("released"));
    }

    #[test]
    fn safetensors_alternate_shapes_accept_only_declared_layouts() {
        let plan = safe_plan(
            vec![SafetensorsTensorConstraint::required(
                "convolution",
                vec![4, 1, 2],
                StoredDtypeConstraint::Floating,
            )
            .with_alternate_shapes([vec![4, 2, 1]])],
            Vec::new(),
            CatalogPolicy::strict(),
        );
        let alternate = BTreeMap::from([safe("convolution", &[4, 2, 1], StoredDtype::BF16)]);
        assert!(validate_catalog(
            &alternate,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        )
        .is_empty());

        let undeclared = BTreeMap::from([safe("convolution", &[4, 2], StoredDtype::BF16)]);
        let issues = validate_catalog(
            &undeclared,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        );
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].kind, CheckpointIssueKind::ShapeMismatch);
    }

    #[test]
    fn element_count_constraints_accept_reshape_equivalent_storage() {
        let plan = safe_plan(
            vec![SafetensorsTensorConstraint::required(
                "convolution",
                vec![4, 1, 2],
                StoredDtypeConstraint::Floating,
            )
            .with_element_count(8)],
            Vec::new(),
            CatalogPolicy::strict(),
        );
        let reshaped = BTreeMap::from([safe("convolution", &[4, 2], StoredDtype::BF16)]);
        assert!(validate_catalog(
            &reshaped,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        )
        .is_empty());

        let wrong = BTreeMap::from([safe("convolution", &[4, 3], StoredDtype::BF16)]);
        let issues = validate_catalog(
            &wrong,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        );
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].kind, CheckpointIssueKind::ShapeMismatch);
        assert!(issues[0].detail.contains("must contain 8 elements"));
    }

    fn alternatives() -> Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>> {
        vec![AlternativeLayoutGroup {
            id: "projection".into(),
            required: true,
            variants: vec![
                LayoutVariant {
                    id: "packed".into(),
                    tensors: vec![SafetensorsTensorConstraint::required(
                        "packed",
                        vec![4, 2],
                        StoredDtypeConstraint::Floating,
                    )],
                    discriminator_keys: vec!["packed".into()],
                },
                LayoutVariant {
                    id: "split".into(),
                    tensors: vec![
                        SafetensorsTensorConstraint::required(
                            "gate",
                            vec![2, 2],
                            StoredDtypeConstraint::Floating,
                        ),
                        SafetensorsTensorConstraint::required(
                            "up",
                            vec![2, 2],
                            StoredDtypeConstraint::Floating,
                        ),
                    ],
                    discriminator_keys: vec!["gate".into(), "up".into()],
                },
            ],
        }]
    }

    #[test]
    fn alternatives_report_missing_partial_and_conflicting_layouts() {
        let plan = safe_plan(Vec::new(), alternatives(), CatalogPolicy::strict());
        let evaluate = |catalog: BTreeMap<_, _>| {
            validate_catalog(
                &catalog,
                &plan.identity,
                &plan.common_tensors,
                &plan.layout_groups,
                &plan.catalog_policy,
            )
        };
        assert!(evaluate(BTreeMap::from([safe("packed", &[4, 2], StoredDtype::F32)])).is_empty());
        let missing = evaluate(BTreeMap::new());
        assert_eq!(missing[0].kind, CheckpointIssueKind::MissingTensor);
        let partial = evaluate(BTreeMap::from([safe("gate", &[2, 2], StoredDtype::F32)]));
        assert!(partial
            .iter()
            .any(|issue| issue.kind == CheckpointIssueKind::MissingTensor));
        let conflict = evaluate(BTreeMap::from([
            safe("packed", &[4, 2], StoredDtype::F32),
            safe("gate", &[2, 2], StoredDtype::F32),
            safe("up", &[2, 2], StoredDtype::F32),
        ]));
        assert!(conflict
            .iter()
            .any(|issue| issue.detail.contains("conflicting variants")));
        let mixed_partial = evaluate(BTreeMap::from([
            safe("packed", &[4, 2], StoredDtype::F32),
            safe("gate", &[2, 2], StoredDtype::F32),
        ]));
        assert!(mixed_partial
            .iter()
            .any(|issue| issue.detail.contains("partially present variants")));
    }

    #[test]
    fn resolution_selects_one_layout_and_only_its_physical_sources() {
        let plan = safe_plan(Vec::new(), alternatives(), CatalogPolicy::strict());
        let catalog = BTreeMap::from([safe("packed", &[4, 2], StoredDtype::BF16)]);
        let resolved = resolve_catalog(
            &catalog,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            resolved.source_keys().iter().cloned().collect::<Vec<_>>(),
            ["packed"]
        );
        assert!(!resolved.source_keys().contains("gate"));
        assert!(!resolved.source_keys().contains("up"));
    }

    #[test]
    fn gguf_operation_class_constraints_are_checked() {
        let constraints = vec![
            GgufTensorConstraint::required(
                "index",
                vec![2],
                GgufTypeConstraint::OperationClass(TensorOperation::I32),
            ),
            GgufTensorConstraint::required(
                "matrix",
                vec![2, 2],
                GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
            ),
            GgufTensorConstraint::required(
                "flexible",
                vec![2, 2],
                GgufTypeConstraint::OperationClass(TensorOperation::Dense),
            )
            .with_alternate_shapes([vec![2, 1, 2]]),
        ];
        let plan =
            GgufCheckpointPlan::new("test", constraints, Vec::new(), CatalogPolicy::strict())
                .unwrap();
        let catalog = BTreeMap::from([
            (
                "index".into(),
                PhysicalMetadata {
                    shape: vec![2],
                    encoding: GgufType::I32,
                },
            ),
            (
                "matrix".into(),
                PhysicalMetadata {
                    shape: vec![2, 2],
                    encoding: GgufType::Q4K,
                },
            ),
            (
                "flexible".into(),
                PhysicalMetadata {
                    shape: vec![2, 1, 2],
                    encoding: GgufType::F16,
                },
            ),
        ]);
        assert!(validate_catalog(
            &catalog,
            &plan.identity,
            &plan.common_tensors,
            &plan.layout_groups,
            &plan.catalog_policy,
        )
        .is_empty());
    }

    #[test]
    fn issue_order_does_not_depend_on_checkpoint_key_order() {
        let plan = safe_plan(
            vec![SafetensorsTensorConstraint::required(
                "required",
                vec![2],
                StoredDtypeConstraint::Exact(StoredDtype::F32),
            )],
            Vec::new(),
            CatalogPolicy::strict(),
        );
        let left = BTreeMap::from([
            safe("z", &[1], StoredDtype::F16),
            safe("a", &[1], StoredDtype::F16),
        ]);
        let right = left
            .iter()
            .rev()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let validate = |catalog: &BTreeMap<_, _>| {
            validate_catalog(
                catalog,
                &plan.identity,
                &plan.common_tensors,
                &plan.layout_groups,
                &plan.catalog_policy,
            )
        };
        assert_eq!(validate(&left), validate(&right));
    }
}
