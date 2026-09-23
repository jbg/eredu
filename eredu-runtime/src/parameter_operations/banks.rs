//! Joins architecture metadata to exact independently stored parameter members.
use super::{PreparedParameterLocation, PreparedParameterSlot};
use crate::ParameterBankKey;
use eredu_checkpoint::recipe::RecipeMetadata;
use eredu_core::parameters::ParameterError;
use eredu_nn::ParameterMetadata;
use std::collections::{BTreeMap, BTreeSet};

/// One physical member of an architecture-declared grouped parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedBankParameterMember {
    /// Exact selected member, including its original global identity.
    pub key: ParameterBankKey,
    /// Native binding name inside this independently stored member.
    pub binding: String,
    /// Architecture-declared full parameter identity.
    pub parameter: String,
    /// Actual output after selected lowering and rank-local transforms.
    pub materialized: RecipeMetadata,
}

/// One complete rank-local parameter and its ordered independent source owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedBankParameter {
    /// Public parameter slot facts, retaining architecture sharing/companion metadata.
    pub slot: PreparedParameterSlot,
    /// Members in ascending global identity order; each contributes one leading row.
    pub members: Vec<PreparedBankParameterMember>,
}

/// Validates a cold join without materializing a member or changing its ownership.
///
/// Gaps in global expert IDs are valid: the selected local bank contains only its
/// owned members. A parameter may not span unrelated bank/unit owners, disagree in
/// encoding geometry, or omit a member that its companions use.
pub fn prepare_bank_parameter_slots(
    declarations: &[ParameterMetadata],
    members: impl IntoIterator<Item = PreparedBankParameterMember>,
) -> Result<Vec<PreparedBankParameter>, ParameterError> {
    let mut declared = BTreeMap::new();
    for declaration in declarations {
        if declared
            .insert(declaration.id.as_str(), declaration)
            .is_some_and(|previous| previous != declaration)
        {
            return Err(ParameterError::Invalid(
                "conflicting parameter declaration".into(),
            ));
        }
    }
    let mut grouped = BTreeMap::<String, Vec<PreparedBankParameterMember>>::new();
    for member in members {
        if member.binding.is_empty()
            || member.materialized.shape.first() != Some(&1)
            || member.materialized.byte_len == 0
        {
            return Err(ParameterError::Invalid(
                "invalid independent parameter member".into(),
            ));
        }
        grouped
            .entry(member.parameter.clone())
            .or_default()
            .push(member);
    }
    let mut output = Vec::with_capacity(grouped.len());
    for (id, mut members) in grouped {
        let parameter = declared
            .get(id.as_str())
            .ok_or_else(|| ParameterError::Missing(id.clone()))?;
        members.sort_unstable_by_key(|member| member.key);
        let first = &members[0];
        let bank = first.key.bank();
        let unit = first.key.unit();
        let mut materialized = first.materialized.clone();
        materialized.shape[0] = members.len();
        materialized.byte_len = 0;
        let mut seen = BTreeSet::new();
        for member in &members {
            if member.key.bank() != bank
                || member.key.unit() != unit
                || !seen.insert(member.key)
                || member.materialized.shape != first.materialized.shape
                || member.materialized.dtype != first.materialized.dtype
                || member.materialized.byte_len != first.materialized.byte_len
            {
                return Err(ParameterError::Invalid(
                    "inconsistent independent parameter members".into(),
                ));
            }
            materialized.byte_len = materialized
                .byte_len
                .checked_add(member.materialized.byte_len)
                .ok_or(ParameterError::Overflow)?;
        }
        output.push(PreparedBankParameter {
            slot: PreparedParameterSlot {
                parameter: (*parameter).clone(),
                materialized,
                backing: None,
                location: PreparedParameterLocation::Bank { bank, unit },
            },
            members,
        });
    }
    for declaration in declarations {
        if let Some(owner) = &declaration.linear_companion_of {
            if output.iter().any(|p| p.slot.parameter.id == *owner)
                && !output.iter().any(|p| p.slot.parameter.id == declaration.id)
            {
                return Err(ParameterError::Missing(declaration.id.as_str().into()));
            }
        }
    }
    for parameter in &output {
        if let Some(owner) = &parameter.slot.parameter.linear_companion_of {
            let owner = output
                .iter()
                .find(|candidate| candidate.slot.parameter.id == *owner)
                .ok_or_else(|| ParameterError::Missing(owner.as_str().into()))?;
            if parameter
                .members
                .iter()
                .map(|m| m.key)
                .ne(owner.members.iter().map(|m| m.key))
            {
                return Err(ParameterError::Invalid(
                    "independent companion ownership differs".into(),
                ));
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::recipe::RecipeDtype;
    use eredu_nn::{LinearCompanionRole, ParameterSpec};

    fn declarations() -> Vec<ParameterMetadata> {
        let mut weight =
            ParameterMetadata::from_spec(&ParameterSpec::trainable("read").unwrap(), true);
        weight.group = Some("routed-parameters".into());
        let mut scale =
            ParameterMetadata::from_spec(&ParameterSpec::trainable("scale").unwrap(), false);
        scale.linear_companion = Some(LinearCompanionRole::Scale);
        scale.linear_companion_of = Some(weight.id.clone());
        vec![weight, scale]
    }

    fn member(id: &str, index: usize) -> PreparedBankParameterMember {
        PreparedBankParameterMember {
            key: ParameterBankKey::new(7, 11, index),
            binding: format!("local-{id}"),
            parameter: id.into(),
            materialized: RecipeMetadata {
                shape: if id == "read" {
                    vec![1, 4, 3]
                } else {
                    vec![1, 2, 1]
                },
                dtype: RecipeDtype::F32,
                byte_len: if id == "read" { 48 } else { 8 },
            },
        }
    }

    #[test]
    fn bank_parameters_preserve_noncontiguous_owners_and_declared_companions() {
        let declarations = declarations();
        let parameters = prepare_bank_parameter_slots(
            &declarations,
            [
                member("scale", 9),
                member("read", 9),
                member("read", 2),
                member("scale", 2),
            ],
        )
        .unwrap();
        assert_eq!(parameters.len(), 2);
        let read = &parameters[0];
        assert_eq!(read.slot.parameter, declarations[0]);
        assert_eq!(read.slot.materialized.shape, [2, 4, 3]);
        assert_eq!(read.slot.materialized.byte_len, 96);
        let loan = read.slot.bank_loan_usage().unwrap();
        assert_eq!(loan.retained_bytes, 192);
        assert_eq!(loan.host_bytes, 1248);
        assert_eq!(
            read.slot.location,
            PreparedParameterLocation::Bank { bank: 7, unit: 11 }
        );
        assert_eq!(
            read.members
                .iter()
                .map(|m| m.key.member())
                .collect::<Vec<_>>(),
            [2, 9]
        );
        assert_eq!(parameters[1].slot.parameter, declarations[1]);
        assert_eq!(parameters[1].slot.materialized.shape, [2, 2, 1]);
    }

    #[test]
    fn bank_parameters_reject_incomplete_companions_and_ambiguous_sources() {
        let declarations = declarations();
        for members in [
            vec![member("read", 2), member("read", 2)],
            vec![member("read", 2), member("read", 9), member("scale", 2)],
            vec![member("read", 2), member("scale", 9)],
            vec![member("missing", 2)],
            vec![member("scale", 2)],
            vec![member("read", 2)],
        ] {
            assert!(prepare_bank_parameter_slots(&declarations, members).is_err());
        }
        for mutate in [
            |m: &mut PreparedBankParameterMember| m.key = ParameterBankKey::new(8, 11, 9),
            |m: &mut PreparedBankParameterMember| m.key = ParameterBankKey::new(7, 12, 9),
            |m: &mut PreparedBankParameterMember| m.materialized.shape[1] = 5,
            |m: &mut PreparedBankParameterMember| m.materialized.dtype = RecipeDtype::F16,
            |m: &mut PreparedBankParameterMember| m.materialized.byte_len = 44,
        ] {
            let mut other = member("read", 9);
            mutate(&mut other);
            assert!(
                prepare_bank_parameter_slots(&declarations, [member("read", 2), other]).is_err()
            );
        }
    }

    #[test]
    fn bank_parameter_byte_sum_rejects_overflow_before_materialization() {
        let mut a = member("read", 2);
        let mut b = member("read", 9);
        a.materialized.byte_len = u64::MAX;
        b.materialized.byte_len = u64::MAX;
        assert!(matches!(
            prepare_bank_parameter_slots(&declarations(), [a, b]),
            Err(ParameterError::Overflow)
        ));
    }

    #[test]
    fn bank_parameters_join_repeated_shared_declarations_without_duplicating_storage() {
        let original = declarations();
        let members = [member("read", 2), member("scale", 2)];
        let expected = prepare_bank_parameter_slots(&original, members.clone()).unwrap();
        let repeated = [original.clone(), original.clone()].concat();
        assert_eq!(
            prepare_bank_parameter_slots(&repeated, members.clone()).unwrap(),
            expected
        );
        let mut conflicting = repeated;
        conflicting[2].trainable = !conflicting[2].trainable;
        assert!(prepare_bank_parameter_slots(&conflicting, members).is_err());
    }
}
