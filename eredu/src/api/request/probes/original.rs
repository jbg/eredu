//! Actual original J/C operations beneath the shared behavioral recognizer.
use super::{Operations, Probe, ifm, inkling, records};
use crate::api::request::ChatTemplateRequest;
use eredu_core::{HostPreparationAuthority, ModelRuntime, TokenInputRejection};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalChatProfileError, OriginalChatProfilePreparation,
    OriginalChatRenderOperationError, OriginalChatTemplate, OriginalEncodedTokenIds,
    OriginalRenderedChat, OriginalTextSourceError, OriginalTokenizer,
};
use eredu_text::chat_storage::{ChatMessageError, ChatMessages, ChatRenderContext};
use eredu_text::tokenizer::structural::{
    StructuralTokenFailure, StructuralTokenSource, resolve_structural_with,
    structural_control_bytes,
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Policy(#[from] crate::api::request::profile::ProfileRequestFailure),
    #[error(transparent)]
    History(#[from] crate::api::request::profile::tagged::Failure),
    #[error(transparent)]
    Ifm(#[from] ifm::ControlError),
    #[error(transparent)]
    Domain(#[from] TokenInputRejection),
    #[error(transparent)]
    Profile(#[from] OriginalChatProfileError),
    #[error(transparent)]
    Messages(#[from] ChatMessageError),
    #[error(transparent)]
    Render(#[from] OriginalChatRenderOperationError),
    #[error(transparent)]
    Encode(#[from] OriginalTextSourceError),
}
/// Actual render/encoding failure retires before its J/C-bound probe account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct Failure {
    #[source]
    cause: Cause,
    completed: Option<OriginalRenderedChat>,
    host: Option<HostPreparationAuthority>,
}
impl Failure {
    fn is_probe_nonmatch(&self) -> bool {
        matches!(&self.cause, Cause::Render(OriginalChatRenderOperationError::Render(cause)) if cause.is_template_rejection())
    }
    fn before(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            completed: None,
            host: None,
        }
    }
    fn held(cause: impl Into<Cause>, host: &HostPreparationAuthority) -> Self {
        Self {
            cause: cause.into(),
            completed: None,
            host: Some(host.clone()),
        }
    }
}

pub(crate) struct Tokens<'a, B: OriginalChatBackend> {
    runtime: &'a ModelRuntime<B>,
    tokenizer: &'a OriginalTokenizer,
}
impl<'a, B: OriginalChatBackend> Tokens<'a, B> {
    pub(crate) fn new(runtime: &'a ModelRuntime<B>, tokenizer: &'a OriginalTokenizer) -> Self {
        Self { runtime, tokenizer }
    }
}
impl<B: OriginalChatBackend> StructuralTokenSource for Tokens<'_, B> {
    type Encoded = OriginalEncodedTokenIds;
    type Error = OriginalTextSourceError;
    fn added_id(&self, spelling: &str) -> Option<u32> {
        self.tokenizer.added_token_id(spelling)
    }
    fn token_id(&self, spelling: &str) -> Option<u32> {
        self.tokenizer.token_id(spelling)
    }
    fn roundtrips(&self, id: u32, spelling: &str) -> bool {
        self.tokenizer.spelling(id) == Some(spelling)
    }
    fn encode(&mut self, spelling: &str) -> Result<Self::Encoded, Self::Error> {
        // This operation owns its E allocation and failure prefix. It is never
        // replaced by a lookup-only assertion that an added token is atomic.
        let encoded = B::encode_original_text_ids(self.runtime, self.tokenizer, spelling, false)?;
        if !encoded.matches_source(self.tokenizer) {
            // A completed CPU encoding has no pending execution. Destroy its
            // real output before returning the fixed foreign-source refusal.
            drop(encoded);
            return Err(TokenInputRejection::IdentityMismatch.into());
        }
        Ok(encoded)
    }
    fn encoded_ids(encoded: &Self::Encoded) -> &[u32] {
        encoded.ids()
    }
}

/// One lexical loan of the actual selected sources. All descriptor births and
/// validation controls are paid before entering their shared workers.
pub(crate) struct Original<'a, B: OriginalChatBackend> {
    runtime: &'a ModelRuntime<B>,
    template: &'a OriginalChatTemplate,
    tokenizer: &'a OriginalTokenizer,
    defaults: Option<&'a serde_json::Map<String, serde_json::Value>>,
    preparation: OriginalChatProfilePreparation,
    host: HostPreparationAuthority,
}
impl<'a, B: OriginalChatBackend> Original<'a, B> {
    pub(crate) fn prepare(
        runtime: &'a ModelRuntime<B>,
        template: &'a OriginalChatTemplate,
        tokenizer: &'a OriginalTokenizer,
        defaults: Option<&'a serde_json::Map<String, serde_json::Value>>,
        capacity: u64,
    ) -> Result<Self, Failure> {
        B::validate_original_chat_sources(runtime, template, tokenizer).map_err(Failure::before)?;
        let preparation = B::prepare_original_chat_profile(runtime, template, tokenizer, capacity)
            .map_err(Failure::before)?;
        if !preparation.has_sources(template, tokenizer) {
            return Err(Failure::before(TokenInputRejection::IdentityMismatch));
        }
        let parts = [
            size_of::<Self>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<Option<inkling::Recognition>, Failure>>(),
            size_of::<(
                &ModelRuntime<B>,
                &OriginalChatTemplate,
                &OriginalTokenizer,
                Option<&serde_json::Map<String, serde_json::Value>>,
                u64,
            )>(),
            super::control_bytes::<Failure>()
                .ok_or_else(|| Failure::before(TokenInputRejection::Overflow))?,
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| Failure::before(TokenInputRejection::Overflow))?;
        let host = preparation
            .reserve_controls(controls)
            .map_err(Failure::before)?;
        Ok(Self {
            runtime,
            template,
            tokenizer,
            defaults,
            preparation,
            host,
        })
    }
    pub(crate) fn recognize_inkling(&mut self) -> Result<Option<inkling::Recognition>, Failure> {
        inkling::recognize(self)
    }
    fn pay(&self, bytes: Option<usize>) -> Result<HostPreparationAuthority, Failure> {
        let bytes =
            bytes.ok_or_else(|| Failure::held(TokenInputRejection::Overflow, &self.host))?;
        self.preparation
            .reserve_controls(bytes)
            .map_err(|cause| Failure::held(cause, &self.host))
    }
}
impl<B: OriginalChatBackend> Operations for Original<'_, B> {
    type Error = Failure;
    fn structural(&mut self, spellings: &[&str]) -> Result<bool, Failure> {
        let controls = structural_control_bytes::<Tokens<'_, B>, &str>()
            .and_then(|n| {
                n.checked_add(size_of::<(
                    Tokens<'_, B>,
                    [u32; super::MAX_STRUCTURAL],
                    Result<
                        (),
                        StructuralTokenFailure<OriginalTextSourceError, OriginalEncodedTokenIds>,
                    >,
                    HostPreparationAuthority,
                )>())
            })
            .and_then(|n| n.checked_add(size_of::<(&mut Self, &[&str])>()));
        let host = self.pay(controls)?;
        let mut result = [0; super::MAX_STRUCTURAL];
        let result = result
            .get_mut(..spellings.len())
            .ok_or_else(|| Failure::held(TokenInputRejection::Overflow, &host))?;
        let mut source = Tokens {
            runtime: self.runtime,
            tokenizer: self.tokenizer,
        };
        match resolve_structural_with(&mut source, spellings, result) {
            Ok(()) => Ok(true),
            Err(StructuralTokenFailure::Encoding { cause, .. }) => Err(Failure::held(cause, &host)),
            Err(StructuralTokenFailure::NonAtomic { encoded, .. }) => {
                if !encoded.matches_source(self.tokenizer) {
                    return Err(Failure::held(TokenInputRejection::IdentityMismatch, &host));
                }
                // The actual completed non-atomic encoding retires here. The
                // cumulative probe control account remains live in self.
                Ok(false)
            }
            Err(_) => Ok(false),
        }
    }
    fn render_matches(
        &mut self,
        probe: Probe,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Failure> {
        match self.render_with_request(probe, None, contains, suffix) {
            Err(cause) if cause.is_probe_nonmatch() => Ok(false),
            result => result,
        }
    }
}
impl<B: OriginalChatBackend> ifm::RequestOperations for Original<'_, B> {
    fn render_request_matches(
        &mut self,
        probe: Probe,
        request: &ChatTemplateRequest,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Failure> {
        self.render_with_request(probe, Some(request), contains, suffix)
    }
    fn control_failure(&self, cause: ifm::ControlError) -> Failure {
        Failure::held(cause, &self.host)
    }
}
impl<B: OriginalChatBackend> Original<'_, B> {
    pub(crate) fn recognize_ifm(
        &mut self,
        request: &ChatTemplateRequest,
    ) -> Result<Option<ifm::Recognition>, Failure> {
        ifm::recognize(self, request)
    }
    pub(crate) fn recognize_remaining(
        &mut self,
    ) -> Result<Option<super::remaining::Recognition>, Failure> {
        super::remaining::recognize(self)
    }
    fn render_with_request(
        &mut self,
        probe: Probe,
        request: Option<&ChatTemplateRequest>,
        contains: &[&str],
        suffix: Option<&str>,
    ) -> Result<bool, Failure> {
        self.inspect_with_request(probe, request, |text| {
            super::matches(text, contains, suffix)
        })
    }
    fn inspect_with_request<T, F: FnOnce(&str) -> T>(
        &mut self,
        probe: Probe,
        request: Option<&ChatTemplateRequest>,
        inspect: F,
    ) -> Result<T, Failure> {
        let parts = [
            records::control_bytes()
                .ok_or_else(|| Failure::held(TokenInputRejection::Overflow, &self.host))?,
            size_of::<(
                Probe,
                records::Input<'_>,
                ChatRenderContext<'_>,
                ChatMessages<'_>,
                OriginalRenderedChat,
            )>(),
            size_of::<Result<OriginalRenderedChat, OriginalChatRenderOperationError>>(),
            size_of::<Result<ChatRenderContext<'_>, ChatMessageError>>(),
            size_of::<(&Self, F)>(),
            size_of::<T>(),
            size_of::<Result<T, Failure>>(),
            size_of::<Option<OriginalRenderedChat>>(),
            size_of::<Option<&ChatTemplateRequest>>(),
            size_of::<Option<ifm::Overrides<'_>>>(),
            size_of::<&[eredu_text::chat_storage::ChatScalarBinding<'_>]>(),
        ];
        let host = self.pay(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add),
        )?;
        let overrides = request.map(ifm::Overrides::new);
        probe.with_records(|input| {
            let context = ChatRenderContext::from_messages(
                ChatMessages::from_records(input.messages)
                    .map_err(|cause| Failure::held(cause, &host))?,
            )
            .with_tools(eredu_text::chat_storage::ChatInputArray::Record(
                input.tools,
            ))
            .with_variables(
                self.defaults,
                request.map(|request| &request.extra_template_kwargs),
            )
            .map_err(|cause| Failure::held(cause, &host))?
            .with_scalar_overrides(
                overrides
                    .as_ref()
                    .map_or(input.kwargs, ifm::Overrides::as_slice),
            )
            .with_clock(eredu_text::chat_storage::chat_clock_snapshot());
            let rendered =
                B::render_original_chat(self.runtime, self.template, self.tokenizer, context)
                    .map_err(|cause| Failure::held(cause, &host))?;
            if !rendered.has_sources(self.template, self.tokenizer) {
                let mut failure = Failure::held(TokenInputRejection::IdentityMismatch, &host);
                failure.completed = Some(rendered);
                return Err(failure);
            }
            B::validate_original_chat_render(self.runtime, &rendered)
                .map_err(|cause| Failure::held(cause, &host))?;
            // The host account above funds this concrete inspection closure and
            // its result. It does not construct an execution consumer.
            Ok(inspect(rendered.prompt(input.generation)))
        })
    }
}

impl<B: OriginalChatBackend> super::remaining::ProtocolOperations for Original<'_, B> {
    fn protocol_facts(
        &mut self,
        mapping: bool,
    ) -> Result<Option<super::remaining::Facts>, Failure> {
        match self.inspect_with_request(
            Probe::Protocol { mapping },
            None,
            super::remaining::Facts::inspect,
        ) {
            Ok(facts) => Ok(Some(facts)),
            Err(cause) if cause.is_probe_nonmatch() => Ok(None),
            Err(cause) => Err(cause),
        }
    }
}

impl<B: OriginalChatBackend> super::selection::Operations for Original<'_, B> {
    fn history_failure(
        &self,
        cause: crate::api::request::profile::tagged::Failure,
        _messages: &[serde_json::Value],
        _profile: &'static str,
    ) -> Failure {
        Failure::held(cause, &self.host)
    }
}
impl<B: OriginalChatBackend> Original<'_, B> {
    pub(crate) fn select(
        &mut self,
        request: &ChatTemplateRequest,
    ) -> Result<super::selection::Selected, Failure> {
        super::selection::select(self, request)
    }
}

/// Fixed selected policy retains the actual source/account used to derive it.
/// No constructor admits caller-provided selection facts or profile identities.
#[derive(Debug)]
pub(crate) struct PolicyOwner {
    profile: crate::runtime::chat::PreparedFormatProfile,
    source: OriginalChatProfilePreparation,
    host: HostPreparationAuthority,
}
impl PolicyOwner {
    pub(crate) fn preparation(&self) -> &OriginalChatProfilePreparation {
        &self.source
    }
    pub(crate) fn profile(&self) -> &crate::runtime::chat::PreparedFormatProfile {
        &self.profile
    }
    pub(crate) fn metadata_funding(&self) -> &eredu_core::HostMetadataFunding {
        self.source.metadata_funding()
    }
    pub(crate) fn generation(&self, requested: bool) -> bool {
        self.profile.generation_prompt_behavior.resolve(requested)
    }
    pub(crate) fn has_sources(
        &self,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
    ) -> bool {
        self.source.has_sources(template, tokenizer)
    }
}
/// Borrowed explicit scalar text ends before the retained selected source owner.
pub(crate) struct PreparedPolicy<'request> {
    bindings: crate::api::request::profile::ProfileRequestBindings<'request>,
    owner: PolicyOwner,
}
impl<'request> PreparedPolicy<'request> {
    pub(crate) fn owner(&self) -> &PolicyOwner {
        &self.owner
    }
    pub(crate) fn metadata_funding(&self) -> &eredu_core::HostMetadataFunding {
        self.owner.metadata_funding()
    }
    pub(crate) fn bindings(&self) -> &[eredu_text::chat_storage::ChatScalarBinding<'request>] {
        self.bindings.as_slice()
    }
    pub(crate) fn into_owner(self) -> PolicyOwner {
        self.owner
    }
}
impl<B: OriginalChatBackend> Original<'_, B> {
    pub(crate) fn prepare_policy<'request>(
        &mut self,
        request: &'request ChatTemplateRequest,
    ) -> Result<PreparedPolicy<'request>, Failure> {
        let parts = [
            crate::runtime::chat::PreparedFormatProfile::control_bytes()
                .ok_or_else(|| Failure::held(TokenInputRejection::Overflow, &self.host))?,
            size_of::<PolicyOwner>(),
            size_of::<PreparedPolicy<'request>>(),
            size_of::<Result<PreparedPolicy<'request>, Failure>>(),
            size_of::<OriginalChatProfilePreparation>(),
            size_of::<HostPreparationAuthority>(),
        ];
        let host = self.pay(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add),
        )?;
        let selection = self.select(request)?;
        let profile = crate::api::request::selected_profile(selection, request);
        let bindings = profile
            .request_bindings(request)
            .map_err(|cause| Failure::held(cause, &host))?;
        Ok(PreparedPolicy {
            bindings,
            owner: PolicyOwner {
                profile,
                source: self.preparation.clone(),
                host,
            },
        })
    }
}
