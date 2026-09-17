//! Fixed request controls derived from the actual recognized format profile.
pub(crate) mod selected;
pub(crate) mod tagged;
use super::{ChatTemplateRequest, TextModelError};
use crate::runtime::chat::{
    PreparedFormatProfile, ReasoningEffortControl, ReasoningTemplateControl,
};
use eredu_text::chat_storage::{ChatScalarBinding, ChatScalarBindingValue};

/// This value is produced from an already recognized profile, not a model name.
/// Its scalar mapping supplies neither protocol recognition nor parser authority.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProfileRequestControls {
    reasoning: ReasoningTemplateControl,
    effort: Option<ReasoningEffortControl>,
    reject_extra_effort: bool,
    muse_strength: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ProfileRequestError {
    #[error("recognized profile does not expose reasoning_effort control")]
    ExtraEffort,
    #[error("reasoning effort must be a string")]
    EffortType,
    #[error("reasoning_effort cannot be combined with enable_thinking=false")]
    DisabledEffort,
    #[error("recognized profile has no explicit reasoning effort control")]
    EffortUnavailable,
    #[error("reasoning effort is outside the recognized profile's declared values")]
    EffortValue,
    #[error("recognized profile does not expose a reasoning-disable control")]
    DisableUnavailable,
    #[error("reasoning strength must be a string")]
    StrengthType,
    #[error("reasoning strength is outside the recognized profile's declared values")]
    StrengthValue,
}
/// Stack-owned lexical overlay. Only its used prefix is borrowed by the renderer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProfileRequestBindings<'a> {
    values: [ChatScalarBinding<'a>; 2],
    len: usize,
}
impl<'a> ProfileRequestBindings<'a> {
    pub(crate) fn as_slice(&self) -> &[ChatScalarBinding<'a>] {
        &self.values[..self.len]
    }
    // Ordinary preparation keeps its existing owned kwargs. The controlled
    // producer borrows as_slice and never calls this allocating adapter.
    pub(super) fn into_owned(self) -> [Option<(String, serde_json::Value)>; 2] {
        std::array::from_fn(|index| {
            (index < self.len).then(|| {
                let binding = self.values[index];
                let value = match binding.value {
                    ChatScalarBindingValue::Bool(value) => serde_json::Value::Bool(value),
                    ChatScalarBindingValue::Text(value) => serde_json::Value::String(value.into()),
                };
                (binding.name.into(), value)
            })
        })
    }
}
impl ProfileRequestControls {
    pub(crate) fn from_profile(profile: &PreparedFormatProfile) -> Self {
        let identity = profile.identity.as_deref();
        Self {
            reasoning: profile.reasoning_template_control,
            effort: profile.reasoning_effort_control,
            reject_extra_effort: identity.is_some_and(|value| value.starts_with("qwen3.6.")),
            muse_strength: identity == Some("muse-glimmer.atem.v1"),
        }
    }
    fn effort_value<'a>(
        &self,
        request: &'a ChatTemplateRequest,
    ) -> Result<Option<&'a str>, ProfileRequestError> {
        if let Some(effort) = &request.reasoning_effort {
            return Ok(Some(effort));
        }
        self.effort
            .and_then(|control| request.extra_template_kwargs.get(control.kwarg))
            .map(|value| value.as_str().ok_or(ProfileRequestError::EffortType))
            .transpose()
    }
    pub(crate) fn validate_effort(
        &self,
        request: &ChatTemplateRequest,
    ) -> Result<(), ProfileRequestError> {
        if self.reject_extra_effort
            && request.reasoning_effort.is_none()
            && request
                .extra_template_kwargs
                .contains_key("reasoning_effort")
        {
            return Err(ProfileRequestError::ExtraEffort);
        }
        if let Some(value) = self.effort_value(request)? {
            if request.enable_thinking == Some(false) {
                return Err(ProfileRequestError::DisabledEffort);
            }
            let control = self.effort.ok_or(ProfileRequestError::EffortUnavailable)?;
            if !control.supported.contains(&value) {
                return Err(ProfileRequestError::EffortValue);
            }
        }
        Ok(())
    }
    pub(crate) fn validate_strength(
        &self,
        request: &ChatTemplateRequest,
    ) -> Result<(), ProfileRequestError> {
        if !self.muse_strength {
            return Ok(());
        }
        if request.enable_thinking == Some(false) {
            return Err(ProfileRequestError::DisableUnavailable);
        }
        if let Some(value) = request.extra_template_kwargs.get("reasoning_strength") {
            let strength = value.as_str().ok_or(ProfileRequestError::StrengthType)?;
            if !matches!(strength, "low" | "medium" | "high" | "xhigh") {
                return Err(ProfileRequestError::StrengthValue);
            }
        }
        Ok(())
    }
    pub(crate) fn bindings<'a>(
        &self,
        request: &'a ChatTemplateRequest,
    ) -> Result<ProfileRequestBindings<'a>, ProfileRequestError> {
        self.validate_effort(request)?;
        self.validate_strength(request)?;
        let mut output = ProfileRequestBindings {
            values: [ChatScalarBinding {
                name: "",
                value: ChatScalarBindingValue::Bool(false),
            }; 2],
            len: 0,
        };
        if let Some(enabled) = request.enable_thinking {
            if !(self.muse_strength
                && request
                    .extra_template_kwargs
                    .contains_key("reasoning_strength"))
            {
                output.values[output.len] = self.reasoning.borrowed_template_entry(enabled);
                output.len += 1;
            }
        }
        if let Some(effort) = &request.reasoning_effort {
            let control = self.effort.ok_or(ProfileRequestError::EffortUnavailable)?;
            output.values[output.len] = ChatScalarBinding {
                name: control.kwarg,
                value: ChatScalarBindingValue::Text(effort),
            };
            output.len += 1;
        }
        Ok(output)
    }
    /// Fixed mapping/validation controls, independent of source string lengths.
    /// Render planning separately prices the actual borrowed scalar contents.
    pub(crate) fn control_bytes() -> usize {
        std::mem::size_of::<(
            Self,
            ProfileRequestError,
            ProfileRequestBindings<'static>,
            &Self,
            &ChatTemplateRequest,
            Option<&str>,
            Result<Option<&str>, ProfileRequestError>,
            Result<(), ProfileRequestError>,
            Result<ProfileRequestBindings<'static>, ProfileRequestError>,
            Option<ReasoningEffortControl>,
            ChatScalarBinding<'static>,
        )>()
    }
}
impl ProfileRequestError {
    // Preserve ordinary diagnostic text while the original path retains this
    // fixed error by value. No formatted strings are required for validation.
    pub(super) fn ordinary(
        self,
        profile: &PreparedFormatProfile,
        request: &ChatTemplateRequest,
    ) -> TextModelError {
        let identity = profile.identity.as_deref().unwrap_or("unregistered");
        let message = match self {
            Self::ExtraEffort => "Qwen3.6 does not expose reasoning_effort control".into(),
            Self::EffortType => format!("{} must be a string for format profile {:?}",
                profile.reasoning_effort_control.expect("same validated control").kwarg, identity),
            Self::DisabledEffort => "reasoning_effort cannot be combined with enable_thinking=false".into(),
            Self::EffortUnavailable => format!("format profile {:?} does not expose reasoning_effort control", identity),
            Self::EffortValue => {
                let control = profile.reasoning_effort_control.expect("same validated control");
                let value = request.reasoning_effort.as_deref().or_else(|| request.extra_template_kwargs.get(control.kwarg).and_then(serde_json::Value::as_str)).expect("same invalid effort");
                format!("unsupported reasoning_effort {value:?} for format profile {:?}; expected {}", identity, control.supported.join(", "))
            }
            Self::DisableUnavailable => "Muse-Glimmer does not expose a reasoning-disable control; enable_thinking=false is not supported".into(),
            Self::StrengthType => "Muse-Glimmer reasoning_strength must be one of low, medium, high, or xhigh".into(),
            Self::StrengthValue => {
                let strength = request.extra_template_kwargs.get("reasoning_strength").and_then(serde_json::Value::as_str).expect("same invalid strength");
                format!("unsupported Muse-Glimmer reasoning_strength {strength:?}; expected low, medium, high, or xhigh")
            }
        };
        TextModelError::ToolConstraint(message)
    }
}
