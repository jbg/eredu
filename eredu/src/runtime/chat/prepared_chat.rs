//! Canonical prepared chat retains its original render and compilation.
use super::{
    CapabilitySupport, ChatCapabilities, GenerationRuntimePlan, NativeToolSupport,
    SemanticRuntimePlan, SemanticSupport,
};
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalControllerCompilation, OriginalRenderedChat, OriginalTextSourceBudget,
    OriginalTokenizer,
};
use eredu_text::tokenizer::ChatTemplateIdentity;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
struct Data {
    render: OriginalRenderedChat,
    generation: bool,
    capacity: u64,
    template_identity: ChatTemplateIdentity,
    policy: crate::api::CompiledChatPolicy,
    compilation: OriginalControllerCompilation,
    // Every rendered/compiled alias retires before this preparation ceiling.
    source_budget: OriginalTextSourceBudget,
}
/// An originally prepared chat with authenticated tokenizer, template and policy.
/// Clones share the same source allocations and preparation ceiling.
#[derive(Debug, Clone)]
pub struct PreparedChat(Option<Arc<Data>>);
impl PreparedChat {
    pub(crate) fn controller_sources(
        &self,
    ) -> eredu_runtime::working_memory::ControllerCompilationSources<'_> {
        use eredu_runtime::working_memory::ControllerCompilationOutput;
        self.data().policy.controller_sources()
    }
    fn data(&self) -> &Data {
        self.0.as_deref().expect("live prepared chat")
    }
    pub(crate) fn render(&self) -> &OriginalRenderedChat {
        &self.data().render
    }
    pub(crate) fn generation(&self) -> bool {
        self.data().generation
    }
    /// Enforced request capacity authenticated by this prepared chat, in bytes.
    pub fn capacity(&self) -> u64 {
        self.data().capacity
    }
    pub(crate) fn tokenizer_source(&self) -> &OriginalTokenizer {
        self.render().tokenizer_source()
    }
    pub(crate) fn compilation(&self) -> &OriginalControllerCompilation {
        &self.data().compilation
    }
    pub(crate) fn publish(
        render: OriginalRenderedChat,
        generation: bool,
        capacity: u64,
        named_entry: Option<&str>,
        policy: crate::api::CompiledChatPolicy,
        compilation: OriginalControllerCompilation,
        source_budget: OriginalTextSourceBudget,
    ) -> Result<Self, PublicationError> {
        let mut data = Data {
            render,
            generation,
            capacity,
            template_identity: ChatTemplateIdentity::Single,
            policy,
            compilation,
            source_budget,
        };
        let result = (|| -> Result<(), Cause> {
            let shell = Layout::new::<[AtomicUsize; 2]>()
                .extend(Layout::new::<Data>())
                .map_err(|_| HostMetadataFundingError::Overflow)?
                .0
                .pad_to_align()
                .size();
            let parts = [
                shell,
                size_of::<Data>(),
                size_of::<Self>(),
                size_of::<Option<Data>>(),
                size_of::<Option<Arc<Data>>>(),
                size_of::<PublicationError>(),
                size_of::<Cause>(),
                size_of::<Result<Self, PublicationError>>(),
                size_of::<Result<(), Cause>>(),
                size_of::<HostMetadataFunding>(),
                size_of::<ChatTemplateIdentity>(),
                size_of::<String>(),
                size_of::<Result<(), std::collections::TryReserveError>>(),
                size_of::<(
                    OriginalRenderedChat,
                    bool,
                    u64,
                    Option<&str>,
                    crate::api::CompiledChatPolicy,
                    OriginalControllerCompilation,
                    OriginalTextSourceBudget,
                )>(),
                HostMetadataFunding::reservation_control_bytes(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(HostMetadataFundingError::Overflow)?;
            let funding = data.compilation.metadata_funding();
            funding.reserve_metadata(bytes)?;
            if let Some(entry) = named_entry {
                funding.reserve_metadata(entry.len())?;
                let mut name = String::new();
                name.try_reserve_exact(entry.len())
                    .map_err(Cause::Allocation)?;
                name.push_str(entry);
                data.template_identity = ChatTemplateIdentity::Named(name);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Self(Some(Arc::new(data)))),
            Err(cause) => Err(PublicationError { cause, data }),
        }
    }
}
impl Drop for PreparedChat {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl PartialEq for PreparedChat {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live prepared chat"),
            other.0.as_ref().expect("live prepared chat"),
        )
    }
}
impl Eq for PreparedChat {}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Allocation(std::collections::TryReserveError),
}
/// Publication failure keeps all original sources and the preparation ceiling.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct PublicationError {
    #[source]
    cause: Cause,
    data: Data,
}

impl PreparedChat {
    /// Returns the rendered prompt.
    pub fn rendered_prompt(&self) -> &str {
        self.data().render.prompt(self.data().generation)
    }

    /// Returns the separately computed generation-prompt contribution.
    pub fn generation_prompt(&self) -> &str {
        self.data().render.generation_suffix()
    }

    /// Returns the selected checkpoint template identity.
    pub fn template_identity(&self) -> &ChatTemplateIdentity {
        &self.data().template_identity
    }

    /// Returns the recognized stable format-protocol identity.
    pub fn format_profile_identity(&self) -> Option<&str> {
        self.data().policy.metadata().profile_identity
    }

    /// Returns native tool capability for the selected template.
    pub fn native_tool_support(&self) -> &NativeToolSupport {
        &self.data().policy.metadata().selection.native_tool_support
    }

    /// Returns semantic response parsing capability for the selected protocol.
    pub fn semantic_support(&self) -> &SemanticSupport {
        &self.data().policy.metadata().selection.semantic_support
    }

    /// Admission for explicit ordinary, speculative or controlled text generation. Tool declarations
    /// (even with `ToolChoice::None`) and required tool calls are rejected.
    /// Explicit thinking requires `allow_unparsed_reasoning`, since text mode
    /// exposes decoded reasoning as ordinary text even on recognized templates.
    /// This reports request admission; backend support for speculation or execution
    /// control is separate, and snapshots require complete estimates and limits.
    pub fn text_generation_support(&self) -> &CapabilitySupport {
        &self
            .data()
            .policy
            .metadata()
            .selection
            .text_generation_support
    }

    /// Returns independently gated protocol capabilities.
    pub fn capabilities(&self) -> &ChatCapabilities {
        &self.data().policy.metadata().selection.capabilities
    }

    #[cfg(test)]
    pub(crate) fn tool_runtime_plan(&self) -> Option<&GenerationRuntimePlan> {
        self.data()
            .policy
            .metadata()
            .generation_runtime_plan
            .as_ref()
            .filter(|plan| plan.has_tool_surface())
    }

    pub(crate) fn semantic_runtime_plan(&self) -> Option<&SemanticRuntimePlan> {
        self.data()
            .policy
            .metadata()
            .generation_runtime_plan
            .as_ref()
            .map(GenerationRuntimePlan::semantic_plan)
    }

    pub(crate) fn generation_runtime_plan(&self) -> Option<&GenerationRuntimePlan> {
        self.data()
            .policy
            .metadata()
            .generation_runtime_plan
            .as_ref()
    }

    /// Returns checkpoint EOS token IDs.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.data().policy.metadata().eos_token_ids
    }

    /// Returns structural token IDs that must survive decoding.
    pub fn preserved_structural_token_ids(&self) -> &[u32] {
        &self.data().policy.metadata().preserved_structural_token_ids
    }

    /// Returns format-profile stop sequences.
    pub fn profile_stop_sequences(&self) -> &[String] {
        &self.data().policy.metadata().stop_sequences
    }
}
