//! Paid immutable slice construction with the exact vocabulary/recipe owner.
use super::{GenerationRuntimePlan, OriginalGrammarVocabulary};
use eredu_nn::workspace::WorkspaceMetadataFundingError;
use llguidance::earley::{SlicerConstructionFailure, SlicerConstructionPlan, SlicerProgram};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("grammar slicer source is unavailable")]
    Source,
    #[error("grammar slicer control geometry overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Construction(#[from] SlicerConstructionFailure),
}
/// Program allocations retire before the exact historical source and funding.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarSlicer {
    program: SlicerProgram,
    vocabulary: OriginalGrammarVocabulary,
}
impl OriginalGrammarSlicer {
    pub(in crate::runtime::chat::constraints) fn program(&self) -> &SlicerProgram {
        &self.program
    }
    pub(in crate::runtime::chat::constraints) fn vocabulary(&self) -> &OriginalGrammarVocabulary {
        &self.vocabulary
    }
    pub(in crate::runtime::chat::constraints) fn matches_plan(
        &self,
        plan: &GenerationRuntimePlan,
    ) -> bool {
        self.vocabulary.matches_plan(plan)
    }
}
/// The owned failed tree/leaf prefix retires before the source's retained H/C.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarSlicerError {
    #[source]
    cause: Cause,
    vocabulary: OriginalGrammarVocabulary,
}
impl OriginalGrammarVocabulary {
    /// Completes immutable slicer storage only. No parser, tokenization callback,
    /// mutable controller or per-step mask is admitted by this constructor.
    pub(in crate::runtime::chat::constraints) fn compile_slicer(
        self,
    ) -> Result<OriginalGrammarSlicer, OriginalGrammarSlicerError> {
        let result = (|| -> Result<SlicerProgram, Cause> {
            let parts = [
                self.recipe
                    .grammar_source_control_bytes()
                    .ok_or(Cause::Overflow)?,
                size_of::<Self>(),
                size_of::<OriginalGrammarSlicer>(),
                size_of::<OriginalGrammarSlicerError>(),
                size_of::<Cause>(),
                size_of::<SlicerProgram>(),
                size_of::<SlicerConstructionPlan<'_>>(),
                size_of::<Result<SlicerConstructionPlan<'_>, SlicerConstructionFailure>>(),
                size_of::<Result<SlicerProgram, SlicerConstructionFailure>>(),
                size_of::<Result<SlicerProgram, Cause>>(),
                size_of::<Result<OriginalGrammarSlicer, OriginalGrammarSlicerError>>(),
                size_of::<Result<(), WorkspaceMetadataFundingError>>(),
                size_of::<&Self>(),
            ];
            self.funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let source = self.recipe.slicer_source().ok_or(Cause::Source)?;
            self.funding.reserve_metadata(
                SlicerConstructionPlan::inspection_control_bytes(source).ok_or(Cause::Overflow)?,
            )?;
            let plan = SlicerConstructionPlan::prepare(source, self.trie.trie())?;
            self.funding
                .reserve_metadata(plan.requirements().required_bytes())?;
            Ok(plan.compile()?)
        })();
        match result {
            Ok(program) => Ok(OriginalGrammarSlicer {
                program,
                vocabulary: self,
            }),
            Err(cause) => Err(OriginalGrammarSlicerError {
                cause,
                vocabulary: self,
            }),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum StepCause {
    #[error("grammar slicer step geometry overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Construction(#[from] llguidance::earley::SlicerStepFailure),
}
/// The exact immutable owner is borrowed until all local destinations retire.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarSlicerStep<'a> {
    step: llguidance::earley::SlicerStep<'a>,
    source: &'a OriginalGrammarSlicer,
}
/// A failed destination prefix keeps the same enclosing source/account loan.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarSlicerStepError<'a> {
    #[source]
    cause: StepCause,
    owner: &'a OriginalGrammarSlicer,
}
impl OriginalGrammarSlicer {
    pub(in crate::runtime::chat::constraints) fn prepare_step(
        &self,
    ) -> Result<OriginalGrammarSlicerStep<'_>, OriginalGrammarSlicerStepError<'_>> {
        let result = (|| -> Result<llguidance::earley::SlicerStep<'_>, StepCause> {
            let parts = [
                size_of::<OriginalGrammarSlicerStep<'_>>(),
                size_of::<OriginalGrammarSlicerStepError<'_>>(),
                size_of::<StepCause>(),
                size_of::<&Self>(),
                size_of::<Result<llguidance::earley::SlicerStep<'_>, StepCause>>(),
                size_of::<Result<OriginalGrammarSlicerStep<'_>, OriginalGrammarSlicerStepError<'_>>>(
                ),
                size_of::<Result<(), WorkspaceMetadataFundingError>>(),
            ];
            self.vocabulary.funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(StepCause::Overflow)?,
            )?;
            self.vocabulary.funding.reserve_metadata(
                self.program
                    .step_inspection_control_bytes()
                    .ok_or(StepCause::Overflow)?,
            )?;
            let plan = self.program.step_plan()?;
            self.vocabulary
                .funding
                .reserve_metadata(plan.requirements().required_bytes())?;
            Ok(plan.compile()?)
        })();
        match result {
            Ok(step) => Ok(OriginalGrammarSlicerStep { step, source: self }),
            Err(cause) => Err(OriginalGrammarSlicerStepError { cause, owner: self }),
        }
    }
}
impl OriginalGrammarSlicerStep<'_> {
    pub(in crate::runtime::chat::constraints) fn compute_bias(
        &mut self,
        rec: &mut llguidance::earley::ParserRecognizer<'_>,
        start: &[u8],
    ) -> &llguidance::toktrie::SimpleVob {
        self.step.compute_bias(rec, start)
    }
    pub(in crate::runtime::chat::constraints) fn matches_plan(
        &self,
        plan: &GenerationRuntimePlan,
    ) -> bool {
        self.source.matches_plan(plan)
    }
}
