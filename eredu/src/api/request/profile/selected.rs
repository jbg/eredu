//! Request policy from the shared behavioral selection, without an owned DTO.
use super::{ProfileRequestBindings, ProfileRequestControls, ProfileRequestError, tagged};
use crate::api::request::{
    ChatTemplateRequest,
    probes::{remaining::Kind, selection::Selected},
};
use crate::runtime::chat::{
    PreparedFormatProfile, ReasoningEffortControl, ReasoningTemplateControl,
    dialect::GenerationPromptBehavior,
};
#[derive(Clone, Copy, Debug)]
pub(crate) struct Policy {
    controls: ProfileRequestControls,
    generation: GenerationPromptBehavior,
    qwen_tagged: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum Failure {
    #[error(transparent)]
    Control(#[from] ProfileRequestError),
    #[error(transparent)]
    History(#[from] tagged::Failure),
}
impl Policy {
    pub(crate) fn from_selected(selected: Selected) -> Self {
        let mut result = Self {
            controls: ProfileRequestControls {
                reasoning: ReasoningTemplateControl::Boolean("enable_thinking"),
                effort: None,
                reject_extra_effort: false,
                muse_strength: false,
            },
            generation: GenerationPromptBehavior::HonorRequest,
            qwen_tagged: false,
        };
        match selected {
            Selected::Muse(_) => {
                result.generation = GenerationPromptBehavior::Always;
                result.controls.reasoning = ReasoningTemplateControl::NamedEffort {
                    kwarg: "reasoning_strength",
                    enabled: "high",
                    disabled: "high",
                };
                result.controls.effort = Some(ReasoningEffortControl {
                    kwarg: "reasoning_strength",
                    supported: &["low", "medium", "high", "xhigh"],
                });
                result.controls.muse_strength = true;
            }
            Selected::Inkling(_) => {
                result.controls.reasoning = ReasoningTemplateControl::NamedEffort {
                    kwarg: "reasoning_effort",
                    enabled: "high",
                    disabled: "none",
                };
            }
            Selected::Ifm { declaration, .. } => {
                result.generation = declaration.generation;
                result.controls.reasoning =
                    ReasoningTemplateControl::Boolean(declaration.reasoning_kwarg);
                result.controls.effort = Some(ReasoningEffortControl {
                    kwarg: "reasoning_effort",
                    supported: &["high", "medium", "low"],
                });
            }
            Selected::Remaining {
                behavior,
                declaration,
            } => {
                result.generation = declaration.generation;
                result.controls.reasoning =
                    ReasoningTemplateControl::Boolean(declaration.reasoning_kwarg);
                result.qwen_tagged = matches!(behavior.kind, Kind::Qwen36 | Kind::Qwen38);
                result.controls.reject_extra_effort = behavior.kind == Kind::Qwen36;
                if behavior.kind == Kind::Qwen38 {
                    result.controls.effort = Some(ReasoningEffortControl {
                        kwarg: "reasoning_effort",
                        supported: &["low", "medium", "xhigh"],
                    });
                }
            }
            Selected::Gemma(_) | Selected::Unregistered => {}
        }
        result
    }
    /// Ordinary DTO construction uses these same selected control declarations.
    pub(crate) fn apply_to(self, profile: &mut PreparedFormatProfile) {
        profile.generation_prompt_behavior = self.generation;
        profile.reasoning_template_control = self.controls.reasoning;
        profile.reasoning_effort_control = self.controls.effort;
    }
    pub(crate) fn generation(self, requested: bool) -> bool {
        self.generation.resolve(requested)
    }
    pub(crate) fn bindings<'a>(
        &self,
        request: &'a ChatTemplateRequest,
    ) -> Result<ProfileRequestBindings<'a>, Failure> {
        // Same ordinary order: effort controls, tagged history, then strength.
        // IFM history already ran in recognition before these controls.
        self.controls.validate_effort(request)?;
        if self.qwen_tagged {
            tagged::validate(&request.messages, "</parameter>")?;
        }
        self.controls.validate_strength(request)?;
        Ok(self.controls.bindings(request)?)
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<Selected>(),
            size_of::<Failure>(),
            size_of::<Result<ProfileRequestBindings<'_>, Failure>>(),
            size_of::<(&Self, &ChatTemplateRequest)>(),
            ProfileRequestControls::control_bytes(),
            tagged::control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
