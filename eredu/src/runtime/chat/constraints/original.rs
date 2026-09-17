//! Shared prepared-plan selection into actual originally funded controllers.
use super::{
    ConstraintController, GenerationRuntimePlan, OriginalPreparedGrammarController,
    forbidden::ForbiddenSourceError,
    grammar_source::{OriginalGrammarStartupError, OriginalPreparedGrammarControllerError},
    selection::{self, Selection},
};
use eredu_core::{
    SharedControllerBytes, SharedTokenFilter, SpeculativeTokenFilterController,
    speculative::{PreparedGrammarController, PreparedGrammarInstallError},
};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::{OriginalChatBackend, OriginalSemanticControllerSource};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Control {
    #[error("original semantic controller startup extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
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
    funding: WorkspaceMetadataFunding,
}
impl OriginalControllerSourceError {
    fn source_cause(&self) -> &(dyn std::error::Error + 'static) {
        match self.failure.as_ref() {
            Some(failure) => failure.as_ref().as_ref().expect("failed controller startup"),
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
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(self.source_cause()) }
}
impl ConstraintController {
    /// The ordinary selector chooses the actual retained controller source.
    /// Native source authentication remains in the existing sampler binding.
    #[inline(never)]
    pub(crate) fn from_original_generation_plan<B: OriginalChatBackend>(
        runtime: &eredu_core::ModelRuntime<B>,
        plan: &GenerationRuntimePlan,
        validity: SharedTokenFilter,
        capacity: usize,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, OriginalControllerSourceError> {
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
                size_of::<WorkspaceMetadataFunding>(),
                size_of::<Result<Self, Cause>>(),
                size_of::<Result<Self, OriginalControllerSourceError>>(),
                size_of::<Result<(), Control>>(),
                size_of::<Result<(), WorkspaceMetadataFundingError>>(),
                size_of::<Result<Self, ForbiddenSourceError>>(),
                size_of::<Result<super::grammar_source::OriginalGrammarState, OriginalGrammarStartupError>>(),
                size_of::<Result<OriginalPreparedGrammarController, OriginalPreparedGrammarControllerError>>(),
                size_of::<Result<Self, PreparedGrammarInstallError<OriginalPreparedGrammarController>>>(),
                size_of::<(&eredu_core::ModelRuntime<B>, &GenerationRuntimePlan,
                    SharedTokenFilter, usize, &WorkspaceMetadataFunding)>(),
            ];
            funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Control::Overflow)?)?;
            Ok(())
        })();
        if let Err(control) = controls {
            return Err(OriginalControllerSourceError {
                failure: None, control: Some(control),
                source: plan.generation_constraint().inner.recipe.source().clone(), funding: funding.clone(),
            });
        }
        let mut failure = Box::new(None);
        match construct::<B>(runtime, plan, validity, capacity, funding) {
            Ok(controller) => Ok(controller),
            Err(cause) => {
                *failure = Some(cause);
                Err(OriginalControllerSourceError {
                    failure: Some(failure), control: None,
                    source: plan.generation_constraint().inner.recipe.source().clone(), funding: funding.clone(),
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
fn construct<B: OriginalChatBackend>(
    runtime: &eredu_core::ModelRuntime<B>, plan: &GenerationRuntimePlan,
    validity: SharedTokenFilter, capacity: usize, funding: &WorkspaceMetadataFunding,
) -> Result<ConstraintController, Cause> {
    match Selection::from_plan(plan)? {
        Selection::Forbidden(_) => Ok(ConstraintController::from_original_forbidden_generation_plan::<B>(
            runtime, plan, validity, capacity, funding,
        )?),
        Selection::Active => {
            let state = plan.generation_constraint().inner.original_grammar_state::<B>(runtime, funding)?;
            let grammar = state.into_controller(capacity, validity)?;
            Ok(ConstraintController::from_prepared_grammar(grammar, funding)?)
        }
        Selection::Auto(_) => {
            let state = plan.generation_constraint().inner.original_grammar_state::<B>(runtime, funding)?;
            let grammar = state.into_auto_controller(capacity, validity)?;
            Ok(ConstraintController::from_prepared_grammar(grammar, funding)?)
        }
    }
}
