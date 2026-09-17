//! Closed behavioral probe operations shared by ordinary and admitted callers.
use super::{ChatTokenizer, ModelChatTemplate};

pub(crate) mod gemma;
pub(crate) mod ifm;
pub(crate) mod inkling;
pub(crate) mod muse;
pub(crate) const MAX_STRUCTURAL: usize = crate::runtime::chat::ifm::TOKENS.len();
pub(crate) mod original;
pub(crate) mod records;
pub(crate) mod remaining;
pub(crate) mod selection;

/// Exact finite inputs used by the shared recognizer. This declaration is not a
/// recognized-profile witness and cannot select a profile by model name.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Probe {
    InklingHistory,
    IfmEffort(ifm::Effort),
    IfmUser,
    IfmTools,
    Protocol { mapping: bool },
    QwenEffort,
    GemmaHistory,
    GemmaPrompt(bool),
    GemmaTool { mapping: bool },
    ReasoningHistory,
    MuseStrength(muse::Strength),
    InklingEffort { enabled: bool },
    Tool { mapping: bool },
}

/// A renderer and encoder loan from the same selected immutable sources.
/// False means the behavior did not match; errors preserve admitted failures.
pub(crate) trait Operations {
    type Error;
    fn structural(&mut self, spellings: &[&str]) -> Result<bool, Self::Error>;
    fn render_matches(
        &mut self,
        probe: Probe,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Self::Error>;
}

/// The same nonallocating output predicate for either completed render owner.
pub(crate) fn matches(rendered: &str, contains: &[&str], suffix: Option<&str>) -> bool {
    contains.iter().all(|part| rendered.contains(part))
        && suffix.is_none_or(|suffix| rendered.ends_with(suffix))
}

/// Actual owned JSON consumed by the ordinary renderer. The original producer
/// must admit these concrete containers before invoking construction and retain
/// that allowance through render/error retirement; this type grants no budget.
pub(crate) struct ProbeInput {
    pub(crate) messages: Vec<serde_json::Value>,
    pub(crate) tools: Vec<serde_json::Value>,
    pub(crate) kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    pub(crate) generation: bool,
}
impl Probe {
    pub(crate) fn construct(self) -> ProbeInput {
        self.with_records(|input| input.into_ordinary())
    }
    pub(super) fn tool_input(arguments: records::Arguments<'_>) -> ProbeInput {
        records::tool(arguments, |input| input.into_ordinary())
    }
}

pub(super) struct Ordinary<'a> {
    pub(super) tokenizer: &'a mut ChatTokenizer,
    pub(super) selected_template: &'a ModelChatTemplate,
    pub(super) model_id: &'a str,
}
impl Ordinary<'_> {
    pub(super) fn render(&mut self, input: ProbeInput) -> Option<String> {
        self.render_result(input).ok().flatten()
    }
    pub(super) fn render_result(
        &mut self,
        input: ProbeInput,
    ) -> Result<Option<String>, super::TextModelError> {
        Ok(self
            .tokenizer
            .apply_chat_template_json(
                self.selected_template.clone(),
                [input.messages],
                Some(&input.tools),
                self.model_id,
                input.generation,
                input.kwargs.as_ref(),
            )?
            .into_iter()
            .next())
    }
}
impl Operations for Ordinary<'_> {
    type Error = std::convert::Infallible;
    fn structural(&mut self, spellings: &[&str]) -> Result<bool, Self::Error> {
        use eredu_text::tokenizer::structural::{
            OrdinaryStructuralTokens, resolve_structural_with,
        };
        // Exact largest declaration of this recognizer; no sparse token extent.
        let mut result = [0; MAX_STRUCTURAL];
        let Some(result) = result.get_mut(..spellings.len()) else {
            return Ok(false);
        };
        Ok(resolve_structural_with(
            &mut OrdinaryStructuralTokens(self.tokenizer),
            spellings,
            result,
        )
        .is_ok())
    }
    fn render_matches(
        &mut self,
        probe: Probe,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Self::Error> {
        Ok(self
            .render(probe.construct())
            .is_some_and(|rendered| matches(&rendered, contains, suffix)))
    }
}

impl remaining::ProtocolOperations for Ordinary<'_> {
    fn protocol_facts(&mut self, mapping: bool) -> Result<Option<remaining::Facts>, Self::Error> {
        Ok(self
            .render(Probe::Protocol { mapping }.construct())
            .as_deref()
            .map(remaining::Facts::inspect))
    }
}

/// Shared finite recognizer and matching frames. JSON construction, rendering
/// and source encoding reserve their own measured storage before execution.
pub(crate) fn control_bytes<E>() -> Option<usize> {
    use std::mem::size_of;
    let parts = [
        muse::control_bytes::<E>()?,
        ifm::control_bytes::<E>()?,
        remaining::control_bytes::<E>()?,
        selection::control_bytes::<E>()?,
        gemma::control_bytes::<E>()?,
        size_of::<Probe>(),
        size_of::<ProbeInput>(),
        size_of::<&str>(),
        size_of::<&[&str]>(),
        size_of::<Option<&str>>(),
        size_of::<std::slice::Iter<'_, &str>>(),
        size_of::<Result<bool, E>>(),
        size_of::<inkling::Recognition>(),
        size_of::<Option<inkling::Recognition>>(),
        size_of::<Result<Option<inkling::Recognition>, E>>(),
        size_of::<[u32; MAX_STRUCTURAL]>(),
        size_of::<[&str; 6]>(),
        size_of::<[&str; 5]>(),
        size_of::<[&str; 3]>(),
        size_of::<[(bool, &str); 2]>(),
        size_of::<std::array::IntoIter<(bool, &str), 2>>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}
