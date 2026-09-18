//! One paid producer for immutable chat policy and its published metadata.
use super::{Rejection, Selection};
use crate::runtime::chat::{
    ChatTemplateRequest, GenerationRuntimePlan, PreparedFormatProfile, ProfileStrings, ToolChoice,
    constraints::{ConstraintCompiler, PreparationFailure},
};
use llguidance::derivre::{ParserAllocationFailure, ParserAllocationFunding, ParserStorageError};
use std::{
    alloc::Layout,
    error::Error,
    fmt,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
pub(crate) struct Metadata {
    pub(crate) selection: Selection,
    pub(crate) profile_identity: Option<&'static str>,
    pub(crate) generation_runtime_plan: Option<GenerationRuntimePlan>,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) preserved_structural_token_ids: Vec<u32>,
    pub(crate) stop_sequences: Vec<String>,
    // Last: published allocations and the closed Arc retire before their payer.
    _funding: ParserAllocationFunding,
}

/// Closed immutable policy aliases never clone grammar or metadata storage.
#[derive(Debug, Clone)]
pub(crate) struct CompiledChatPolicy(Option<Arc<Metadata>>);
impl eredu_runtime::working_memory::ControllerCompilationOutput for CompiledChatPolicy {
    fn controller_sources(&self) -> eredu_runtime::working_memory::ControllerCompilationSources<'_> {
        self.metadata().generation_runtime_plan.as_ref()
            .map_or_else(Default::default, GenerationRuntimePlan::controller_sources)
    }
}
impl CompiledChatPolicy {
    pub(crate) fn metadata(&self) -> &Metadata {
        self.0.as_deref().expect("live compiled policy")
    }
    pub(crate) fn compile<E: fmt::Display, F>(
        profile: &PreparedFormatProfile,
        request: &ChatTemplateRequest,
        eos: &[u32],
        compiler: Result<Option<&ConstraintCompiler>, &str>,
        funding: &ParserAllocationFunding,
        resolve: F,
    ) -> Result<Self, Failure<E>>
    where
        F: FnOnce(&[String], &ParserAllocationFunding) -> Result<Vec<u32>, E>,
    {
        let result = (|| -> Result<Self, Cause<E>> {
            let shell = Layout::new::<[AtomicUsize; 2]>()
                .extend(Layout::new::<Metadata>())
                .map_err(|_| Cause::Storage(funding.storage_overflow()))?
                .0
                .pad_to_align();
            let controls = [
                shell.size(),
                size_of::<Metadata>(),
                size_of::<Self>(),
                size_of::<Option<Arc<Metadata>>>(),
                size_of::<Option<Metadata>>(),
                size_of::<Failure<E>>(),
                size_of::<Cause<E>>(),
                size_of::<Result<Self, Failure<E>>>(),
                size_of::<Result<Self, Cause<E>>>(),
                size_of::<Result<Vec<u32>, E>>(),
                size_of::<F>(),
                size_of::<(
                    &PreparedFormatProfile,
                    &ChatTemplateRequest,
                    &[u32],
                    Result<Option<&ConstraintCompiler>, &str>,
                    &ParserAllocationFunding,
                )>(),
                size_of::<Result<GenerationRuntimePlan, PreparationFailure>>(),
                size_of::<Vec<String>>(),
                size_of::<Vec<u32>>(),
                size_of::<Option<GenerationRuntimePlan>>(),
                size_of::<Layout>(),
                Selection::control_bytes(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or_else(|| Cause::Storage(funding.storage_overflow()))?;
            funding.reserve(bytes).map_err(Cause::Funding)?;
            let selection = Selection::prepare(profile, request, matches!(compiler, Ok(Some(_))))
                .map_err(Cause::Policy)?;
            let generation_runtime_plan = match selection.runtime {
                Some(runtime) => {
                    let compiler = match compiler {
                        Ok(Some(compiler)) => compiler,
                        Ok(None) => return Err(Cause::MissingCompiler),
                        Err(error) => {
                            return Err(Cause::Compiler(
                                funding.try_copy_str(error).map_err(Cause::Storage)?,
                            ));
                        }
                    };
                    let spellings = strings(runtime.structural_tokens, funding)?;
                    let ids = resolve(&spellings, funding).map_err(Cause::Structural)?;
                    Some(
                        compiler
                            .compile_generation_plan(
                                runtime.dialect,
                                runtime.parameters,
                                if runtime.has_tool_surface {
                                    &request.tools
                                } else {
                                    &[]
                                },
                                if runtime.has_tool_surface {
                                    request.tool_choice
                                } else {
                                    ToolChoice::None
                                },
                                request.parallel_tool_calls,
                                spellings,
                                ids,
                                strings(profile.stop_sequences, funding)?,
                                runtime.has_tool_surface,
                            )
                            .map_err(Cause::Declaration)?,
                    )
                }
                None => None,
            };
            let mut preserved_structural_token_ids = Vec::new();
            if let Some(plan) = &generation_runtime_plan {
                let tokens = plan.semantic_plan().structural_tokens();
                funding
                    .try_grow_vec(&mut preserved_structural_token_ids, tokens.len())
                    .map_err(Cause::Storage)?;
                preserved_structural_token_ids.extend(tokens.map(|(id, _)| id));
            }
            let mut eos_token_ids = Vec::new();
            funding
                .try_extend_copy(&mut eos_token_ids, eos)
                .map_err(Cause::Storage)?;
            let stop_sequences = strings(profile.stop_sequences, funding)?;
            Ok(Self(Some(Arc::new(Metadata {
                selection,
                profile_identity: profile.identity,
                generation_runtime_plan,
                eos_token_ids,
                preserved_structural_token_ids,
                stop_sequences,
                _funding: funding.clone(),
            }))))
        })();
        result.map_err(|cause| Failure {
            cause,
            funding: funding.clone(),
        })
    }
}
impl Drop for CompiledChatPolicy {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl PartialEq for CompiledChatPolicy {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (self.metadata(), other.metadata());
        a.profile_identity == b.profile_identity
            && a.selection.native_tool_support == b.selection.native_tool_support
            && a.selection.semantic_support == b.selection.semantic_support
            && a.selection.text_generation_support == b.selection.text_generation_support
            && a.selection.capabilities == b.selection.capabilities
            && a.generation_runtime_plan == b.generation_runtime_plan
            && a.eos_token_ids == b.eos_token_ids
            && a.preserved_structural_token_ids == b.preserved_structural_token_ids
            && a.stop_sequences == b.stop_sequences
    }
}
impl Eq for CompiledChatPolicy {}

fn strings<E>(
    source: ProfileStrings,
    funding: &ParserAllocationFunding,
) -> Result<Vec<String>, Cause<E>> {
    let controls = [
        size_of::<(ProfileStrings, &ParserAllocationFunding)>(),
        size_of::<Vec<String>>(),
        size_of::<String>(),
        size_of::<Result<String, ParserStorageError>>(),
        size_of::<Result<Vec<String>, Cause<E>>>(),
        size_of_val(&source.iter()),
    ];
    let bytes = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or_else(|| Cause::Storage(funding.storage_overflow()))?;
    funding.reserve(bytes).map_err(Cause::Funding)?;
    let mut output = Vec::new();
    funding
        .try_grow_vec(&mut output, source.iter().count())
        .map_err(Cause::Storage)?;
    for text in source.iter() {
        output.push(funding.try_copy_str(text).map_err(Cause::Storage)?);
    }
    Ok(output)
}

#[derive(Debug)]
enum Cause<E> {
    Policy(Rejection),
    MissingCompiler,
    Compiler(String),
    Funding(ParserAllocationFailure),
    Storage(ParserStorageError),
    Structural(E),
    Declaration(PreparationFailure),
}
#[derive(Debug)]
pub(crate) struct Failure<E> {
    cause: Cause<E>,
    funding: ParserAllocationFunding,
}
impl<E: fmt::Display> fmt::Display for Failure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Policy(e) => e.fmt(f),
            Cause::MissingCompiler => {
                f.write_str("the loaded model does not have tokenizer constraint data")
            }
            Cause::Compiler(e) => f.write_str(e),
            Cause::Funding(e) => e.fmt(f),
            Cause::Storage(e) => e.fmt(f),
            Cause::Structural(e) => e.fmt(f),
            Cause::Declaration(e) => e.fmt(f),
        }
    }
}
impl<E: fmt::Display> Failure<E> {
    pub(crate) fn into_ordinary(self) -> crate::api::TextModelError {
        let Self { cause, funding } = self;
        match cause {
            Cause::Declaration(cause) => crate::api::ConstraintError::preparation(cause).into(),
            cause => {
                crate::api::TextModelError::ToolConstraint(Self { cause, funding }.to_string())
            }
        }
    }
}
impl<E: Error + 'static> Error for Failure<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.cause {
            Cause::Policy(e) => Some(e),
            Cause::MissingCompiler | Cause::Compiler(_) => None,
            Cause::Funding(e) => Some(e),
            Cause::Storage(e) => Some(e),
            Cause::Structural(e) => Some(e),
            Cause::Declaration(e) => Some(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{convert::Infallible, sync::atomic::Ordering};

    #[derive(Debug, thiserror::Error)]
    #[error("policy preparation funding refused")]
    struct Refused;

    fn funding(
        refuse_at: usize,
    ) -> (
        ParserAllocationFunding,
        Arc<AtomicUsize>,
        std::sync::Weak<()>,
    ) {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = calls.clone();
        let funding = ParserAllocationFunding::prepare(move |_| {
            let _ = &owner;
            let index = callback_calls.fetch_add(1, Ordering::Relaxed);
            if index == refuse_at {
                Err(Refused)
            } else {
                Ok(())
            }
        })
        .unwrap();
        (funding, calls, weak)
    }

    fn compile(
        funding: &ParserAllocationFunding,
    ) -> Result<CompiledChatPolicy, Failure<Infallible>> {
        let mut profile = crate::runtime::chat::prepare_format_profile("unrecognized template");
        profile.stop_sequences = ProfileStrings::new(&["stop", "終わり"]);
        CompiledChatPolicy::compile(
            &profile,
            &ChatTemplateRequest::default(),
            &[9, 41, 103],
            Ok(None),
            funding,
            |_, _| unreachable!("unknown text profile needs no structural encoding"),
        )
    }

    #[test]
    fn published_metadata_aliases_share_storage_and_retire_the_original_payer() {
        let (funding, _, owner) = funding(usize::MAX);
        let policy = compile(&funding).unwrap();
        assert_eq!(policy.metadata().eos_token_ids, [9, 41, 103]);
        assert_eq!(policy.metadata().stop_sequences, ["stop", "終わり"]);
        assert!(
            policy
                .metadata()
                .selection
                .text_generation_support
                .is_supported()
        );
        assert!(policy.metadata().generation_runtime_plan.is_none());
        let alias = policy.clone();
        assert!(std::ptr::eq(policy.metadata(), alias.metadata()));
        drop((policy, funding));
        assert!(owner.upgrade().is_some());
        assert_eq!(alias.metadata().eos_token_ids, [9, 41, 103]);
        drop(alias);
        assert!(owner.upgrade().is_none());
    }

    #[test]
    fn every_reached_metadata_refusal_keeps_its_payer_and_original_error() {
        let (baseline, calls, _) = funding(usize::MAX);
        drop(compile(&baseline).unwrap());
        let reached = calls.load(Ordering::Relaxed);
        assert!(reached > 3);
        for refuse_at in 1..reached {
            let (funding, calls, owner) = funding(refuse_at);
            let error = compile(&funding).unwrap_err();
            assert_eq!(calls.load(Ordering::Relaxed), refuse_at + 1);
            assert_eq!(error.to_string(), "policy preparation funding refused");
            let mut cause: &(dyn Error + 'static) = &error;
            while let Some(next) = cause.source() {
                cause = next;
            }
            assert!(cause.is::<Refused>());
            drop(funding);
            assert!(owner.upgrade().is_some());
            drop(error);
            assert!(owner.upgrade().is_none());
        }
    }
}
