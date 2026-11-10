//! Cold executable parameter selection for independently owned prediction modules.
use super::{invalid, prediction_topology};
use crate::configuration::{PredictionExtensionPlan, PredictionScopes, SafetensorsModelConfig};
use eredu_core::{artifact::ArtifactError, ParallelRankTopology};
use eredu_runtime::{ComponentDefinition, LocalModelLayout, ReplicatedTextMaterializationTask};
use std::{collections::HashMap, sync::Arc};

/// Exact prediction construction and placement retained before backend allocation.
#[derive(Debug, Clone)]
pub struct SelectedPredictionParameters {
    pub(super) extension: PredictionExtensionPlan,
    pub(super) model: SafetensorsModelConfig,
    pub(super) scopes: PredictionScopes,
    pub(super) layout: Arc<LocalModelLayout>,
    pub(super) rank: ParallelRankTopology,
    pub(super) tasks: Vec<ReplicatedTextMaterializationTask>,
}
impl SelectedPredictionParameters {
    pub(crate) fn new(
        extension: &PredictionExtensionPlan,
        tasks: &[ReplicatedTextMaterializationTask],
        topology: ParallelRankTopology,
    ) -> Result<Self, ArtifactError> {
        let rank = prediction_topology(extension, topology)?;
        let source = extension.definition();
        let members = source.parameters().members_by_target();
        let mut formats = members
            .values()
            .filter_map(|member| {
                member
                    .logical_linear_format()
                    .map(|format| (member.target().to_owned(), format.encoding()))
            })
            .collect::<HashMap<_, _>>();
        for task in tasks {
            if formats.contains_key(task.name()) {
                formats.insert(task.name().to_owned(), task.executable());
            }
        }
        let quantized = || {
            formats
                .iter()
                .filter_map(|(name, format)| {
                    format
                        .weight_quantization()
                        .map(|format| (name.clone(), format))
                })
                .collect()
        };
        let (model, scopes) = match extension.complete_architecture().model() {
            SafetensorsModelConfig::DeepSeekV3(args) => {
                let target = crate::deepseek::v3_with_checkpoint_formats(
                    args,
                    formats.into_iter().collect(),
                )
                .map_err(invalid)?;
                let definition = crate::deepseek::v3::project_parameters(&target, source)
                    .map_err(|error| invalid(error.to_string()))?;
                (
                    SafetensorsModelConfig::DeepSeekV3(target),
                    PredictionScopes::Component(definition),
                )
            }
            SafetensorsModelConfig::DeepSeekV4(args) => {
                let target = crate::deepseek::v4_with_checkpoint_formats(
                    args,
                    formats.into_iter().collect(),
                )
                .map_err(invalid)?;
                let definition = crate::deepseek::v4::project_parameters(&target, source)
                    .map_err(|error| invalid(error.to_string()))?;
                (
                    SafetensorsModelConfig::DeepSeekV4(target),
                    PredictionScopes::Component(definition),
                )
            }
            SafetensorsModelConfig::Inkling(args) => {
                let target =
                    crate::inkling::with_checkpoint_formats(args, quantized()).map_err(invalid)?;
                let definition = crate::inkling::model::project_parameters(&target, source)
                    .map_err(|error| invalid(error.to_string()))?;
                (
                    SafetensorsModelConfig::Inkling(target),
                    PredictionScopes::Component(definition),
                )
            }
            SafetensorsModelConfig::QwenHybrid(args) => {
                let mut target = args.clone();
                let definition = if args.vision.is_some() {
                    target =
                        crate::replicated_text::qwen_hybrid_composite_with_formats(args, formats)
                            .map_err(invalid)?;
                    crate::qwen::hybrid::project_conditional_parameters(&target, source)
                } else {
                    let mut policy = target.text.policy().clone();
                    policy.quantization = None;
                    policy.fp8 = None;
                    policy.linear_formats = formats;
                    target.text = target
                        .text
                        .with_formats(policy)
                        .map_err(|e| invalid(e.to_string()))?;
                    target
                        .text
                        .validate()
                        .map_err(|error| invalid(error.to_string()))?;
                    crate::qwen::hybrid::project_parameters(&target.text, source)
                }
                .map_err(|error| invalid(error.to_string()))?;
                (
                    SafetensorsModelConfig::QwenHybrid(target),
                    PredictionScopes::Component(definition),
                )
            }
            SafetensorsModelConfig::NemotronH(args) => {
                let target = crate::nemotron_h::with_checkpoint_formats(args, quantized())
                    .map_err(invalid)?;
                let declaration = extension
                    .nemotron_declaration()
                    .ok_or_else(|| invalid("Nemotron prediction has no retained typed scope"))?
                    .with_formats(target.clone())
                    .map_err(|error| invalid(error.to_string()))?;
                (
                    SafetensorsModelConfig::NemotronH(target),
                    PredictionScopes::Nemotron(Arc::new(declaration)),
                )
            }
            _ => {
                return Err(invalid(
                    "prediction family has no executable parameter projection",
                ))
            }
        };
        let definition = scopes.definition();
        eredu_runtime::validate_executable_component_projection(source, definition, tasks)
            .map_err(|error| invalid(error.to_string()))?;
        let layout = eredu_runtime::derive_partitioned_local_layout(definition.parameters(), rank)
            .map_err(invalid)?;
        Ok(Self {
            extension: extension.clone(),
            model,
            scopes,
            layout: Arc::new(layout),
            rank,
            tasks: tasks.to_vec(),
        })
    }
    pub(crate) fn same_admission(&self, extension: &PredictionExtensionPlan) -> bool {
        self.extension.same_admission(extension)
    }

    pub(crate) fn routed_parameters(&self) -> std::collections::BTreeSet<String> {
        self.scopes
            .definition()
            .parameters()
            .groups()
            .iter()
            .filter(|group| {
                matches!(
                    group.group().role(),
                    eredu_runtime::ParameterRole::ExpertIntermediate
                        | eredu_runtime::ParameterRole::ExpertOutput
                )
            })
            .flat_map(|group| group.members())
            .map(|member| member.target().to_owned())
            .collect()
    }
    /// Executable formats with source logical identities and ownership retained.
    pub fn definition(&self) -> &Arc<ComponentDefinition> {
        self.scopes.definition()
    }
    /// Exact tensor-rank placement, replicated across target pipeline/expert ranks.
    pub fn layout(&self) -> &Arc<LocalModelLayout> {
        &self.layout
    }
    /// Selected prediction parameter and companion outputs.
    pub fn tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        &self.tasks
    }
}
