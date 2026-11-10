//! Cold physical-module ownership projected from retained parameter declarations.
use super::*;
use std::collections::BTreeMap;

use eredu_nn::DeclareLogicalParameters;
use eredu_runtime::{AuxiliaryModuleResidency, ParameterGroupOwner};

/// Physical identities used by cold admission and the consuming construction driver.
#[derive(Clone, Copy)]
pub(crate) enum PredictionModuleNamespace {
    Embedded,
    Qwen4Exp,
}

pub(crate) fn module_residency(
    namespace: PredictionModuleNamespace,
    ordinal: Option<usize>,
) -> Result<AuxiliaryModuleResidency, eredu_core::artifact::ArtifactError> {
    let id = match (namespace, ordinal) {
        (PredictionModuleNamespace::Embedded, Some(ordinal)) => {
            format!("prediction.unit.{ordinal}")
        }
        (PredictionModuleNamespace::Embedded, None) => "prediction.shared".into(),
        (PredictionModuleNamespace::Qwen4Exp, Some(ordinal)) => {
            format!("qwen4_exp.prediction.depth.{ordinal}")
        }
        (PredictionModuleNamespace::Qwen4Exp, None) => "qwen4_exp.prediction.shared".into(),
    };
    AuxiliaryModuleResidency::new(id, ordinal.is_none()).map_err(|error| invalid(error.to_string()))
}

pub(crate) fn parameter_residencies(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
    depth: usize,
    definition: &eredu_runtime::ComponentDefinition,
) -> Result<BTreeMap<String, AuxiliaryModuleResidency>, eredu_core::artifact::ArtifactError> {
    let declared = definition.parameters().members_by_target();
    let mut residencies = BTreeMap::new();
    let mut assign = |member: &eredu_runtime::ParameterMemberSpec,
                      ordinal: Option<usize>|
     -> Result<(), eredu_core::artifact::ArtifactError> {
        if declared.get(member.target()).copied() != Some(member) {
            return Err(invalid(format!(
                "prediction module parameter {:?} differs from its retained declaration",
                member.target(),
            )));
        }
        let owner = module_residency(PredictionModuleNamespace::Embedded, ordinal)?;
        if residencies
            .insert(member.target().to_owned(), owner)
            .is_some()
        {
            return Err(invalid(format!(
                "prediction parameter {:?} has duplicate physical module ownership",
                member.target(),
            )));
        }
        Ok(())
    };

    if let SafetensorsModelConfig::Inkling(args) = architecture.model() {
        // Placement pins the complete prediction role; bounded materialization
        // leases each typed depth independently and retains only the chain norm.
        let parameters = args.prediction();
        if parameters.depths.len() != depth {
            return Err(invalid(
                "Inkling physical depths differ from the retained extension",
            ));
        }
        let scope = |spec: &dyn DeclareLogicalParameters| {
            eredu_runtime::logical_parameter_group(
                "model.mtp",
                eredu_runtime::ParameterRole::Replicated,
                None,
                spec,
                |_| eredu_runtime::MemberSharding::Replicated,
            )
            .map_err(|error| invalid(error.to_string()))
        };
        for (ordinal, depth) in parameters.depths.iter().enumerate() {
            for member in scope(depth)?.members() {
                assign(member, Some(ordinal))?;
            }
        }
        if let Some(norm) = &parameters.chain_norm {
            for member in scope(norm)?.members() {
                assign(member, None)?;
            }
        }
    } else {
        if definition.prediction_groups().len() != depth {
            return Err(invalid(
                "prediction groups differ from the retained extension depth",
            ));
        }
        let mut units = BTreeMap::new();
        for group in definition.prediction_groups() {
            let index = definition
                .parameters()
                .graph()
                .group_index(group)
                .ok_or_else(|| invalid("prediction residency has no declared execution group"))?;
            for unit in 0..definition
                .unit_count(index)
                .map_err(|error| invalid(error.to_string()))?
            {
                let ordinal = units.len();
                units.insert((group.as_str(), unit), ordinal);
            }
        }
        for group in definition.parameters().groups() {
            let ordinal = match group.owner() {
                ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                } => {
                    let Some(&ordinal) = units.get(&(group.as_str(), *global_unit)) else {
                        continue;
                    };
                    Some(ordinal)
                }
                ParameterGroupOwner::StaticRole(role)
                | ParameterGroupOwner::StaticUnitConsumers { role, .. }
                    if role == "mtp" =>
                {
                    None
                }
                _ => continue,
            };
            for member in group.members() {
                assign(member, ordinal)?;
            }
        }
    }

    // Every declared prediction resource is assigned once. Shared target
    // embeddings/readout roles remain with their target storage owner.
    for group in definition.parameters().groups() {
        let prediction = match group.owner() {
            ParameterGroupOwner::ExecutionUnit { group, .. } => definition
                .prediction_groups()
                .iter()
                .any(|candidate| candidate == group.as_str()),
            ParameterGroupOwner::StaticRole(role)
            | ParameterGroupOwner::StaticUnitConsumers { role, .. } => role == "mtp",
            _ => false,
        };
        if prediction {
            for member in group.members() {
                if !residencies.contains_key(member.target()) {
                    return Err(invalid(format!(
                        "prediction parameter {:?} has no physical module owner",
                        member.target(),
                    )));
                }
            }
        }
    }
    if residencies.is_empty() {
        return Err(invalid(
            "prediction extension has no declared parameter residency",
        ));
    }
    Ok(residencies)
}
