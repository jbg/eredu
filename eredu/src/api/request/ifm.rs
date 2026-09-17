//! Recognition from rendered IFM protocol behavior and tokenizer properties.
use super::{ChatTemplateRequest, TextModelError, probes};

use probes::{Operations, Probe};

pub(super) struct Ordinary<'a>(pub(super) probes::Ordinary<'a>);
impl Operations for Ordinary<'_> {
    type Error = TextModelError;
    fn structural(&mut self, spellings: &[&str]) -> Result<bool, Self::Error> {
        match self.0.structural(spellings) {
            Ok(value) => Ok(value),
            Err(never) => match never {},
        }
    }
    fn render_matches(
        &mut self,
        probe: Probe,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Self::Error> {
        match self.0.render_matches(probe, contains, suffix) {
            Ok(value) => Ok(value),
            Err(never) => match never {},
        }
    }
}
impl probes::ifm::RequestOperations for Ordinary<'_> {
    fn render_request_matches(
        &mut self,
        probe: Probe,
        request: &ChatTemplateRequest,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Self::Error> {
        use eredu_text::chat_storage::ChatScalarBindingValue;
        let mut input = probe.construct();
        let mut kwargs = request.extra_template_kwargs.clone();
        let replacements = probes::ifm::Overrides::new(request);
        for binding in replacements.as_slice() {
            let value = match binding.value {
                ChatScalarBindingValue::Bool(value) => serde_json::Value::Bool(value),
                ChatScalarBindingValue::Text(value) => serde_json::Value::String(value.into()),
            };
            kwargs.insert(binding.name.into(), value);
        }
        input.kwargs = Some(kwargs);
        Ok(self
            .0
            .render_result(input)?
            .is_some_and(|text| probes::matches(&text, contains, suffix)))
    }
    fn control_failure(&self, cause: probes::ifm::ControlError) -> Self::Error {
        TextModelError::ToolConstraint(cause.to_string())
    }
}

impl probes::remaining::ProtocolOperations for Ordinary<'_> {
    fn protocol_facts(
        &mut self,
        mapping: bool,
    ) -> Result<Option<probes::remaining::Facts>, Self::Error> {
        match probes::remaining::ProtocolOperations::protocol_facts(&mut self.0, mapping) {
            Ok(value) => Ok(value),
            Err(never) => match never {},
        }
    }
}
impl probes::selection::Operations for Ordinary<'_> {
    fn history_failure(
        &self,
        cause: super::profile::tagged::Failure,
        messages: &[serde_json::Value],
        profile: &'static str,
    ) -> Self::Error {
        cause.ordinary(messages, profile)
    }
}
