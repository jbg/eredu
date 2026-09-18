//! Shared prepared-plan selection into actual originally funded controllers.
use super::{
    forbidden::ForbiddenSourceError,
    grammar_source::{OriginalGrammarStartupError, OriginalPreparedGrammarControllerError},
    selection::{self, Selection},
    ConstraintController, GenerationRuntimePlan, OriginalPreparedGrammarController,
};
use eredu_core::{
    speculative::{PreparedGrammarController, PreparedGrammarInstallError},
    SharedControllerBytes, SharedTokenFilter, SpeculativeTokenFilterController,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalControllerCompilation, OriginalSemanticControllerSource,
    OriginalTokenizer,
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Control {
    #[error("original semantic controller startup extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Selection(#[from] selection::Error),
    #[error(transparent)]
    Forbidden(#[from] ForbiddenSourceError),
    #[error(transparent)]
    Grammar(#[from] OriginalGrammarStartupError),
    #[error(transparent)]
    Controller(#[from] OriginalPreparedGrammarControllerError),
    #[error(transparent)]
    Install(#[from] PreparedGrammarInstallError<OriginalPreparedGrammarController>),
}
/// Every owning constructor error is placed in the cell paid before startup.
/// This keeps native request error transports small without erasing the cause.
#[derive(Debug)]
pub(crate) struct OriginalControllerSourceError {
    failure: Option<Box<Option<Cause>>>,
    control: Option<Control>,
    source: SharedControllerBytes,
    compilation: OriginalControllerCompilation,
    funding: HostMetadataFunding,
}
impl OriginalControllerSourceError {
    fn source_cause(&self) -> &(dyn std::error::Error + 'static) {
        match self.failure.as_ref() {
            Some(failure) => failure
                .as_ref()
                .as_ref()
                .expect("failed controller startup"),
            None => self.control.as_ref().expect("refused startup controls"),
        }
    }
}
impl std::fmt::Display for OriginalControllerSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.source_cause(), f)
    }
}
impl std::error::Error for OriginalControllerSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source_cause())
    }
}
impl ConstraintController {
    /// The ordinary selector chooses the actual retained controller source.
    /// Native source authentication remains in the existing sampler binding.
    #[inline(never)]
    pub(crate) fn from_original_generation_plan<B: OriginalChatBackend>(
        runtime: &eredu_core::ModelRuntime<B>,
        tokenizer: &OriginalTokenizer,
        plan: &GenerationRuntimePlan,
        compilation: &OriginalControllerCompilation,
        validity: SharedTokenFilter,
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalControllerSourceError> {
        Self::from_original_generation_plan_with(
            plan,
            compilation,
            validity,
            capacity,
            funding,
            |validity| {
                Self::from_original_forbidden_generation_plan::<B>(
                    runtime, tokenizer, plan, validity, capacity, funding,
                )
            },
        )
    }

    /// Shared controller construction, with only the exact tokenizer source
    /// compiler supplied by its enclosing runtime (or neutral test source).
    pub(super) fn from_original_generation_plan_with<F>(
        plan: &GenerationRuntimePlan,
        compilation: &OriginalControllerCompilation,
        validity: SharedTokenFilter,
        capacity: usize,
        funding: &HostMetadataFunding,
        forbidden: F,
    ) -> Result<Self, OriginalControllerSourceError>
    where
        F: FnOnce(SharedTokenFilter) -> Result<Self, ForbiddenSourceError>,
    {
        let controls = (|| -> Result<(), Control> {
            let parts = [
                Selection::control_bytes().ok_or(Control::Overflow)?,
                eredu_core::HostMetadataFunding::reservation_control_bytes(),
                size_of::<Option<Cause>>(),
                size_of::<Box<Option<Cause>>>(),
                size_of::<Option<Box<Option<Cause>>>>(),
                size_of::<Cause>(),
                size_of::<Control>(),
                size_of::<Option<Control>>(),
                size_of::<OriginalControllerSourceError>(),
                size_of::<Self>(),
                size_of::<SharedControllerBytes>(),
                size_of::<HostMetadataFunding>(),
                size_of::<Result<Self, Cause>>(),
                size_of::<Result<Self, OriginalControllerSourceError>>(),
                size_of::<Result<(), Control>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<Result<Self, ForbiddenSourceError>>(),
                size_of::<
                    Result<
                        super::grammar_source::OriginalGrammarState,
                        OriginalGrammarStartupError,
                    >,
                >(),
                size_of::<
                    Result<
                        OriginalPreparedGrammarController,
                        OriginalPreparedGrammarControllerError,
                    >,
                >(),
                size_of::<
                    Result<Self, PreparedGrammarInstallError<OriginalPreparedGrammarController>>,
                >(),
                size_of::<(
                    F,
                    &GenerationRuntimePlan,
                    &OriginalControllerCompilation,
                    SharedTokenFilter,
                    usize,
                    &HostMetadataFunding,
                )>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Control::Overflow)?,
            )?;
            Ok(())
        })();
        if let Err(control) = controls {
            return Err(OriginalControllerSourceError {
                failure: None,
                control: Some(control),
                source: plan.generation_constraint().inner.recipe.source().clone(),
                compilation: compilation.clone(),
                funding: funding.clone(),
            });
        }
        let mut failure = Box::new(None);
        match construct(plan, compilation, validity, capacity, funding, forbidden) {
            Ok(controller) => Ok(controller),
            Err(cause) => {
                *failure = Some(cause);
                Err(OriginalControllerSourceError {
                    failure: Some(failure),
                    control: None,
                    source: plan.generation_constraint().inner.recipe.source().clone(),
                    compilation: compilation.clone(),
                    funding: funding.clone(),
                })
            }
        }
    }
    pub(crate) fn original_semantic_inputs(&self) -> Option<OriginalSemanticControllerSource<'_>> {
        if let Some(source) = self.original_forbidden_inputs() {
            return Some(OriginalSemanticControllerSource::Forbidden(source));
        }
        Some(OriginalSemanticControllerSource::Grammar(
            self.prepared_grammar()?.prepared_grammar_source(),
        ))
    }
}
#[inline(never)]
fn construct<F>(
    plan: &GenerationRuntimePlan,
    compilation: &OriginalControllerCompilation,
    validity: SharedTokenFilter,
    capacity: usize,
    funding: &HostMetadataFunding,
    forbidden: F,
) -> Result<ConstraintController, Cause>
where
    F: FnOnce(SharedTokenFilter) -> Result<ConstraintController, ForbiddenSourceError>,
{
    match Selection::from_plan(plan)? {
        Selection::Forbidden(_) => Ok(forbidden(validity)?),
        Selection::Active => {
            let state = plan
                .generation_constraint()
                .inner
                .original_grammar_state(compilation, funding)?;
            let grammar = state.into_controller(capacity, validity)?;
            Ok(ConstraintController::from_prepared_grammar(
                grammar, funding,
            )?)
        }
        Selection::Auto(_) => {
            let state = plan
                .generation_constraint()
                .inner
                .original_grammar_state(compilation, funding)?;
            let grammar = state.into_auto_controller(capacity, validity)?;
            Ok(ConstraintController::from_prepared_grammar(
                grammar, funding,
            )?)
        }
    }
}
