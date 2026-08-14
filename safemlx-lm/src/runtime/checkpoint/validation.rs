//! Deterministic header-only evaluation of declarative checkpoint plans.

use std::collections::{BTreeMap, BTreeSet};

use safemlx::ops::{GgufCheckpoint, GgufType};

use super::{
    contract::{
        missing, shape_mismatch, CheckpointIssue, CheckpointIssueKind, CheckpointValidation,
    },
    schema::{
        AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorRequirement, TensorRole,
    },
    store::{SafetensorsWeightStore, StoredDtype, WeightMetadata, WeightStore, WeightStoreError},
};

#[derive(Debug, Clone, Eq, PartialEq)]
struct PhysicalMetadata<E> {
    shape: Vec<usize>,
    encoding: E,
}

trait Constraint<E> {
    fn key(&self) -> &str;
    fn aliases(&self) -> &[String];
    fn shape(&self) -> &[usize];
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
        match &self.encoding {
            GgufTypeConstraint::Exact(encoding) => format!("{encoding:?}"),
            GgufTypeConstraint::OneOf(encodings) => format!("one of {encodings:?}"),
            GgufTypeConstraint::OperationClass(operation) => format!("a {operation:?} operation"),
        }
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
pub(crate) fn validate_safetensors_plan(
    store: &SafetensorsWeightStore,
    plan: &SafetensorsCheckpointPlan,
) -> CheckpointValidation {
    let mut catalog = BTreeMap::new();
    let mut metadata_issues = Vec::new();
    for key in store.keys() {
        match store.metadata(&key) {
            Ok(WeightMetadata {
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

/// Validates a GGUF catalog without decoding tensor payloads.
pub(crate) fn validate_gguf_plan(
    checkpoint: &GgufCheckpoint,
    plan: &GgufCheckpointPlan,
) -> CheckpointValidation {
    let catalog = checkpoint
        .catalog()
        .tensors()
        .map(|tensor| {
            let descriptor = tensor.descriptor();
            (
                descriptor.name.clone(),
                PhysicalMetadata {
                    shape: descriptor
                        .mlx_shape()
                        .into_iter()
                        .map(|dimension| usize::try_from(dimension).unwrap_or(usize::MAX))
                        .collect(),
                    encoding: descriptor.ggml_type,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    CheckpointValidation::from_issues(validate_catalog(
        &catalog,
        &plan.identity,
        &plan.common_tensors,
        &plan.layout_groups,
        &plan.catalog_policy,
    ))
}

/// Validates that architecture-supplied tensor pairs use identical encodings.
pub(crate) fn validate_matching_gguf_encodings(
    checkpoint: &GgufCheckpoint,
    pairs: impl IntoIterator<Item = (String, String)>,
    label: &str,
) -> Vec<CheckpointIssue> {
    let catalog = checkpoint
        .catalog()
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
        if gate.descriptor().ggml_type != up.descriptor().ggml_type
            || gate.affine() != up.affine()
            || gate.is_mxfp4() != up.is_mxfp4()
        {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::CompanionMismatch,
                detail: format!(
                    "{label} paired expert tensors {gate_name:?} and {up_name:?} use incompatible encodings {:?} and {:?}",
                    gate.descriptor().ggml_type,
                    up.descriptor().ggml_type
                ),
                tensor_name: Some(gate_name),
                tensor_type_code: Some(gate.descriptor().ggml_type.code()),
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
    variant: &super::schema::LayoutVariant<T>,
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
        && (actual.shape != constraint.shape() || !constraint.accepts(&actual.encoding))
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

    if actual.shape != constraint.shape() {
        issues.push(shape_mismatch(
            actual_key,
            constraint.shape(),
            &actual.shape,
        ));
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

fn companion_issue(name: &str, detail: String) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::CompanionMismatch,
        detail,
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: Some("quantization".into()),
    }
}

fn metadata_failure(name: &str, error: WeightStoreError) -> CheckpointIssue {
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
    use crate::runtime::checkpoint::schema::{LayoutVariant, TensorOperation};

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
    fn gguf_exact_and_operation_class_constraints_are_checked() {
        let constraints = vec![
            GgufTensorConstraint::required(
                "index",
                vec![2],
                GgufTypeConstraint::OneOf(vec![GgufType::I32]),
            ),
            GgufTensorConstraint::required(
                "matrix",
                vec![2, 2],
                GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
            )
            .companion(),
            GgufTensorConstraint::required(
                "optional",
                vec![1],
                GgufTypeConstraint::Exact(GgufType::F32),
            )
            .optional(),
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
