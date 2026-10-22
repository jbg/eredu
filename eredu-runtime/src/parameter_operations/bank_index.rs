//! Exact physical binding addresses and canonical replacement row projections.
use super::PreparedBankParameterMember;
use crate::ParameterBankKey;
use std::collections::BTreeMap;

/// Ambiguous retained bank parameter ownership.
#[derive(Debug, thiserror::Error)]
pub enum BankParameterIndexError {
    /// A physical binding is declared more than once.
    #[error("bank member {key:?} repeats binding {binding:?}")]
    DuplicateBinding {
        /// Global physical storage identity.
        key: ParameterBankKey,
        /// Repeated local binding name.
        binding: String,
    },
    /// One canonical parameter claims two rows in the same physical member.
    #[error("bank parameter {parameter:?} repeats physical member {key:?}")]
    DuplicateParameterMember {
        /// Canonical parameter identity.
        parameter: String,
        /// Repeated physical storage identity.
        key: ParameterBankKey,
    },
}

/// Validated binding metadata and replacement row for one physical member.
/// The row is a dense projection into the canonical parameter's replacement;
/// it never replaces the member's global storage identity.
#[derive(Debug, Clone, Copy)]
pub struct BankParameterMemberAddress<'a> {
    member: &'a PreparedBankParameterMember,
    row: usize,
}
impl<'a> BankParameterMemberAddress<'a> {
    /// Exact retained physical binding and canonical destination.
    pub const fn member(&self) -> &'a PreparedBankParameterMember {
        self.member
    }
    /// Dense replacement row, ordered by global physical member identity.
    pub const fn row(&self) -> usize {
        self.row
    }
}

/// Immutable index shared by parameter discovery and compact replacement access.
/// Members retain source order; row projections use canonical physical-key order,
/// matching full parameter assembly even for sparse selections and unsorted inputs.
#[derive(Debug, Default)]
pub struct PreparedBankParameterIndex {
    members: Vec<PreparedBankParameterMember>,
    bindings: BTreeMap<ParameterBankKey, BTreeMap<String, usize>>,
    rows: Vec<usize>,
}
impl PreparedBankParameterIndex {
    /// Consumes final member metadata and rejects ambiguous physical ownership.
    pub fn new(members: Vec<PreparedBankParameterMember>) -> Result<Self, BankParameterIndexError> {
        let mut bindings = BTreeMap::<_, BTreeMap<_, _>>::new();
        let mut parameters = BTreeMap::<_, BTreeMap<_, _>>::new();
        for (index, member) in members.iter().enumerate() {
            if bindings
                .entry(member.key)
                .or_default()
                .insert(member.binding.clone(), index)
                .is_some()
            {
                return Err(BankParameterIndexError::DuplicateBinding {
                    key: member.key,
                    binding: member.binding.clone(),
                });
            }
            if parameters
                .entry(member.parameter.as_str())
                .or_default()
                .insert(member.key, index)
                .is_some()
            {
                return Err(BankParameterIndexError::DuplicateParameterMember {
                    parameter: member.parameter.clone(),
                    key: member.key,
                });
            }
        }
        let mut rows = vec![0; members.len()];
        for members in parameters.values() {
            for (row, index) in members.values().enumerate() {
                rows[*index] = row;
            }
        }
        Ok(Self {
            members,
            bindings,
            rows,
        })
    }
    /// Original exact member declarations in retained binding order.
    pub fn members(&self) -> &[PreparedBankParameterMember] {
        &self.members
    }
    /// Resolves a global storage member and local binding without source work.
    pub fn binding(
        &self,
        key: ParameterBankKey,
        binding: &str,
    ) -> Option<BankParameterMemberAddress<'_>> {
        let index = *self.bindings.get(&key)?.get(binding)?;
        Some(BankParameterMemberAddress {
            member: &self.members[index],
            row: self.rows[index],
        })
    }
    /// Consumes the index for canonical parameter slot assembly.
    pub fn into_members(self) -> Vec<PreparedBankParameterMember> {
        self.members
    }
}
