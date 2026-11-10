//! Source-free task binding for declared physical prediction modules.
use super::residency::{module_residency, PredictionModuleNamespace};
use super::*;
use eredu_core::residency::OffloadUnitId;
use eredu_runtime::AuxiliaryModuleResidency;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct PreparedPredictionBindings {
    pub(super) tasks: Vec<ReplicatedTextMaterializationTask>,
    pub(super) role: PredictionModuleRole,
}

pub(super) struct PredictionBindingIndex {
    namespace: PredictionModuleNamespace,
    modules: BTreeMap<OffloadUnitId, PreparedPredictionBindings>,
}

impl PredictionBindingIndex {
    pub(super) fn new<'a>(
        namespace: PredictionModuleNamespace,
        declarations: impl IntoIterator<Item = (&'a str, &'a AuxiliaryModuleResidency)>,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<Self, eredu_core::artifact::ArtifactError> {
        let mut owners = BTreeMap::new();
        let mut groups = BTreeMap::<_, (&AuxiliaryModuleResidency, BTreeSet<&str>)>::new();
        for (name, owner) in declarations {
            if owners.insert(name, owner).is_some() {
                return Err(invalid(format!(
                    "duplicate prediction binding identity {name:?}"
                )));
            }
            let (previous, members) = groups
                .entry(owner.module().clone())
                .or_insert_with(|| (owner, BTreeSet::new()));
            if *previous != owner {
                return Err(invalid(
                    "prediction module has conflicting residency declarations",
                ));
            }
            members.insert(name);
        }
        if groups.is_empty() {
            return Err(invalid("prediction has no declared binding modules"));
        }
        let mut modules = groups
            .iter()
            .map(|(id, (owner, _))| {
                (
                    id.clone(),
                    PreparedPredictionBindings {
                        tasks: Vec::new(),
                        role: if owner.shared() {
                            PredictionModuleRole::Shared
                        } else {
                            PredictionModuleRole::Unit
                        },
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut covered = BTreeSet::new();
        for task in tasks {
            let owner = owners.get(task.name()).copied().ok_or_else(|| {
                invalid(format!(
                    "undeclared prediction materialization task {:?}",
                    task.name()
                ))
            })?;
            if task.auxiliary_residency() != Some(owner) {
                return Err(invalid(format!(
                    "prediction task {:?} differs from its declared physical owner",
                    task.name()
                )));
            }
            for name in std::iter::once(task.name()).chain(
                task.output_companions()
                    .iter()
                    .map(ReplicatedTextOutputCompanion::name),
            ) {
                if owners.get(name).copied() != Some(owner) {
                    return Err(invalid(format!(
                        "prediction output {name:?} is not declared in its producer's physical module"
                    )));
                }
                if !covered.insert(name) {
                    return Err(invalid(format!(
                        "prediction output {name:?} has duplicate materialization producers"
                    )));
                }
            }
            modules
                .get_mut(owner.module())
                .expect("declared prediction module is indexed")
                .tasks
                .push(task.clone());
        }
        for (id, (_, members)) in groups {
            let missing = members.difference(&covered).collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(invalid(format!(
                    "declared prediction module {id:?} has no exact materialization tasks for {missing:?}"
                )));
            }
        }
        Ok(Self { namespace, modules })
    }

    pub(super) fn for_extension(
        extension: &PredictionExtensionPlan,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<Self, eredu_core::artifact::ArtifactError> {
        Self::new(
            PredictionModuleNamespace::Embedded,
            extension
                .parameter_residencies()
                .iter()
                .map(|(name, owner)| (name.as_str(), owner)),
            tasks,
        )
    }

    pub(super) fn for_qwen4(
        declaration: &eredu_runtime::ArchitectureParameterDescription,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<Self, eredu_core::artifact::ArtifactError> {
        let mut owners = BTreeMap::new();
        for group in declaration.groups() {
            let ordinal = match group.owner() {
                eredu_runtime::ParameterGroupOwner::StaticRole(role)
                    if role == "prediction_shared" =>
                {
                    None
                }
                eredu_runtime::ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                } if group.as_str() == crate::qwen4_exp::mtp::PREDICTION_EXECUTION_GROUP => {
                    Some(*global_unit)
                }
                _ => {
                    return Err(invalid(
                        "Qwen4 prediction binding has an undeclared module owner",
                    ));
                }
            };
            let owner = module_residency(PredictionModuleNamespace::Qwen4Exp, ordinal)?;
            for member in group.members() {
                if owners.insert(member.target(), owner.clone()).is_some() {
                    return Err(invalid(
                        "Qwen4 prediction binding identity has duplicate owners",
                    ));
                }
            }
        }
        Self::new(
            PredictionModuleNamespace::Qwen4Exp,
            owners.iter().map(|(name, owner)| (*name, owner)),
            tasks,
        )
    }

    pub(super) fn take(
        &mut self,
        ordinal: Option<usize>,
    ) -> Result<PreparedPredictionBindings, eredu_core::artifact::ArtifactError> {
        let owner = module_residency(self.namespace, ordinal)?;
        self.modules.remove(owner.module()).ok_or_else(|| {
            invalid(format!(
                "prediction module {:?} has no unconsumed declared bindings",
                owner.module(),
            ))
        })
    }

    pub(super) fn finish(self) -> Result<(), eredu_core::artifact::ArtifactError> {
        if !self.modules.is_empty() {
            return Err(invalid(format!(
                "prediction construction omitted declared modules {:?}",
                self.modules.keys().collect::<Vec<_>>()
            )));
        }
        Ok(())
    }
}
