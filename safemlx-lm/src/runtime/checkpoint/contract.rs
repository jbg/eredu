//! Structured, format-neutral checkpoint diagnostics.

use crate::error::Error;

/// Stable checkpoint validation categories used by inspection and strict load.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum CheckpointIssueKind {
    MissingTensor,
    UnexpectedTensor,
    ConflictingLayout,
    ShapeMismatch,
    UnsupportedEncoding,
    CompanionMismatch,
    InvalidGeometry,
    ValidationUnavailable,
}

impl CheckpointIssueKind {
    /// Compatibility spelling retained for existing structural call sites.
    #[allow(non_upper_case_globals)]
    pub(crate) const QuantizationCompanionMismatch: Self = Self::CompanionMismatch;
}

/// One structured checkpoint diagnostic.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct CheckpointIssue {
    pub(crate) kind: CheckpointIssueKind,
    pub(crate) detail: String,
    pub(crate) tensor_name: Option<String>,
    pub(crate) tensor_type_code: Option<u32>,
    pub(crate) metadata_key: Option<String>,
}

/// Result of exact, fail-closed checkpoint validation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum CheckpointValidation {
    Exact,
    Invalid(Vec<CheckpointIssue>),
    Unverified(CheckpointIssue),
}

impl CheckpointValidation {
    pub(crate) fn into_loader_result(self) -> Result<(), Error> {
        match self {
            Self::Exact => Ok(()),
            Self::Unverified(issue) => Err(Error::StrictLoadValidation {
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
                Err(Error::StrictLoadValidation { missing, unused })
            }
        }
    }

    pub(crate) fn with_strict_catalog(self, strict: bool) -> Self {
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

    pub(crate) fn from_issues(issues: Vec<CheckpointIssue>) -> Self {
        if issues.is_empty() {
            Self::Exact
        } else {
            Self::Invalid(issues)
        }
    }
}

pub(crate) fn missing(name: &str) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::MissingTensor,
        detail: format!("checkpoint is missing required tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

pub(crate) fn shape_mismatch(name: &str, expected: &[usize], actual: &[usize]) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::ShapeMismatch,
        detail: format!("tensor {name:?} expected shape {expected:?}, got {actual:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}
