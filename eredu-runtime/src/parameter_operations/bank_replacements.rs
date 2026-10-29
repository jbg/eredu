//! Publication-free replacement accounting for independently stored bank members.
use super::PreparedBankParameterMember;
use crate::ParameterBankKey;
use eredu_nn::TensorElementType;
use std::collections::{BTreeMap, BTreeSet};

/// Backend-observed geometry of an already admitted replacement tensor.
/// Reading these facts does not allocate, acquire or publish a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankParameterReplacementFacts {
    /// Actual leading member dimension, absent for a scalar.
    pub leading_dimension: Option<i32>,
    /// Actual scalar representation, absent when the mechanism cannot report it.
    pub element_type: Option<TensorElementType>,
    /// Actual complete replacement payload, including every retained member.
    pub byte_len: u64,
}

/// Invalid replacement geometry or independent member accounting.
#[derive(Debug, thiserror::Error)]
pub enum BankParameterReplacementError {
    /// The replacement cannot supply exactly one floating row per retained member.
    #[error(
        "bank replacement {parameter:?} is incompatible with {members} retained members: {facts:?}"
    )]
    Geometry {
        /// Canonical parameter identity.
        parameter: String,
        /// Independently retained member count.
        members: usize,
        /// Observed replacement facts.
        facts: BankParameterReplacementFacts,
    },
    /// A retained parameter refers to storage absent from the physical catalog.
    #[error("bank replacement {parameter:?} refers to uncataloged member {key:?}")]
    MissingMember {
        /// Canonical parameter identity.
        parameter: String,
        /// Missing physical storage identity.
        key: ParameterBankKey,
    },
    /// Replacing one member payload underflows or overflows its admitted extent.
    #[error("bank replacement {parameter:?} byte accounting overflowed for member {key:?}")]
    ByteOverflow {
        /// Canonical parameter identity.
        parameter: String,
        /// Physical storage whose payload cannot be represented.
        key: ParameterBankKey,
    },
}

/// Complete replacement identities and extents, ready for an outer transaction.
/// This plan owns no tensor and grants no publication or execution authority.
pub struct PreparedBankParameterReplacements {
    parameters: BTreeSet<String>,
    member_bytes: BTreeMap<ParameterBankKey, u64>,
}
impl PreparedBankParameterReplacements {
    /// Moves selected canonical identities and actual independent member charges
    /// into the backend's infallible publication step.
    pub fn into_parts(self) -> (BTreeSet<String>, BTreeMap<ParameterBankKey, u64>) {
        (self.parameters, self.member_bytes)
    }
}

/// Prepares a pool's replacement accounting without mutating any live resource.
/// Every retained member is charged separately, even when source parameters are
/// shared. The callback observes each participating canonical replacement once.
/// Inactive overlays restore the original catalog without inspecting candidates.
pub fn prepare_bank_parameter_replacements(
    members: &[PreparedBankParameterMember],
    catalog: &BTreeMap<ParameterBankKey, u64>,
    active: bool,
    mut replacement: impl FnMut(&str) -> Option<BankParameterReplacementFacts>,
) -> Result<PreparedBankParameterReplacements, BankParameterReplacementError> {
    let mut member_bytes = catalog.clone();
    let mut parameters = BTreeSet::new();
    if active {
        let mut replacements = BTreeMap::new();
        for member in members {
            replacements
                .entry(member.parameter.as_str())
                .or_insert((0usize, None))
                .0 += 1;
        }
        for member in members {
            let (count, facts) = replacements
                .get_mut(member.parameter.as_str())
                .expect("counted parameter");
            // Cache missing replacements as well as present ones.
            let Some(facts) = *facts.get_or_insert_with(|| replacement(&member.parameter)) else {
                continue;
            };
            let count = *count;
            if facts.leading_dimension != i32::try_from(count).ok()
                || !matches!(
                    facts.element_type,
                    Some(TensorElementType::F32 | TensorElementType::F16 | TensorElementType::Bf16)
                )
                || !facts.byte_len.is_multiple_of(count as u64)
            {
                return Err(BankParameterReplacementError::Geometry {
                    parameter: member.parameter.clone(),
                    members: count,
                    facts,
                });
            }
            let bytes = member_bytes.get_mut(&member.key).ok_or_else(|| {
                BankParameterReplacementError::MissingMember {
                    parameter: member.parameter.clone(),
                    key: member.key,
                }
            })?;
            *bytes = bytes
                .checked_sub(member.materialized.byte_len)
                .and_then(|bytes| bytes.checked_add(facts.byte_len / count as u64))
                .ok_or_else(|| BankParameterReplacementError::ByteOverflow {
                    parameter: member.parameter.clone(),
                    key: member.key,
                })?;
        }
        parameters.extend(
            replacements
                .into_iter()
                .filter_map(|(name, (_, facts))| facts.flatten().map(|_| name.to_owned())),
        );
    }
    Ok(PreparedBankParameterReplacements {
        parameters,
        member_bytes,
    })
}
