//! Paid actual lexical input attached to the same original grammar/slicer owner.
mod active;
mod controller;
mod copy;
mod forcing;
pub(crate) use controller::OriginalPreparedGrammarController;
pub(in crate::runtime::chat::constraints) use controller::OriginalPreparedGrammarControllerError;
mod startup;
use super::{OriginalGrammarSlicer, OriginalGrammarVocabulary};
pub(in crate::runtime::chat::constraints) use active::{
    OriginalGrammarState, OriginalGrammarStateConstructionError, OriginalGrammarStateCopyError,
    OriginalGrammarStateError,
};
pub(in crate::runtime::chat::constraints) use copy::OriginalGrammarTokenParserCopyError;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
pub(in crate::runtime::chat::constraints) use forcing::{
    OriginalGrammarForcedTokens, OriginalGrammarForcingError,
};
use llguidance::toktrie;
use llguidance::{
    derivre::{AlphabetCopyFailure, AlphabetCopyPlan, AlphabetInfo},
    earley::lexerspec::{
        LexerInputFailure, LexerRootPlan, LexerRootSource, LexerSpec, RegexVectorInput,
    },
};
pub(in crate::runtime::chat::constraints) use startup::OriginalGrammarStartupError;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("original lexical input control geometry overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Source(#[from] super::super::declaration::Cause),
    #[error(transparent)]
    Roots(#[from] LexerInputFailure),
    #[error(transparent)]
    Alphabet(#[from] AlphabetCopyFailure),
}
/// Actual lexical data, plus the complete immutable grammar/slicer source.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarLexerInputs {
    input: RegexVectorInput,
    source: OriginalGrammarSlicer,
}
/// Failed destination/source prefixes retire before source/funding custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarLexerInputError {
    #[source]
    cause: Cause,
    roots: Option<LexerRootSource>,
    source: OriginalGrammarSlicer,
}
impl OriginalGrammarSlicer {
    pub(in crate::runtime::chat::constraints) fn compile_lexer_inputs(
        self,
    ) -> Result<OriginalGrammarLexerInputs, OriginalGrammarLexerInputError> {
        let mut roots = None;
        let result = (|| -> Result<RegexVectorInput, Cause> {
            let funding = &self.vocabulary().funding;
            let parts = [
                super::super::declaration::HistoricalGrammarDeclaration::inspection_control_bytes()
                    .ok_or(Cause::Overflow)?,
                LexerSpec::root_source_inspection_control_bytes().ok_or(Cause::Overflow)?,
                AlphabetInfo::source_inspection_control_bytes().ok_or(Cause::Overflow)?,
                size_of::<Self>(),
                size_of::<OriginalGrammarLexerInputs>(),
                size_of::<OriginalGrammarLexerInputError>(),
                size_of::<Cause>(),
                size_of::<LexerRootPlan<'_>>(),
                size_of::<LexerRootSource>(),
                size_of::<Option<LexerRootSource>>(),
                size_of::<RegexVectorInput>(),
                size_of::<AlphabetCopyPlan<'_>>(),
                size_of::<Result<LexerRootPlan<'_>, LexerInputFailure>>(),
                size_of::<Result<LexerRootSource, LexerInputFailure>>(),
                size_of::<Result<AlphabetCopyPlan<'_>, AlphabetCopyFailure>>(),
                size_of::<Result<RegexVectorInput, LexerInputFailure>>(),
                size_of::<Result<RegexVectorInput, Cause>>(),
                size_of::<Result<OriginalGrammarLexerInputs, OriginalGrammarLexerInputError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<Result<usize, super::super::declaration::Cause>>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let spec = self.vocabulary().compiled_declaration().lexer_spec();
            let plan = spec.root_source_plan()?;
            funding.reserve_metadata(plan.requirements().required_bytes())?;
            roots = Some(plan.compile()?);
            funding.reserve_metadata(self.vocabulary().declaration.source_copy_control_bytes()?)?;
            let plan = AlphabetInfo::source_copy_plan(
                spec.regex_builder.exprset(),
                roots.as_ref().expect("root source").roots(),
            )?;
            funding.reserve_metadata(plan.requirements().required_bytes())?;
            let (alpha, expressions, normalized) = plan.compile()?;
            Ok(roots
                .take()
                .expect("root source")
                .finish(alpha, expressions, normalized)?)
        })();
        match result {
            Ok(input) => Ok(OriginalGrammarLexerInputs {
                input,
                source: self,
            }),
            Err(cause) => Err(OriginalGrammarLexerInputError {
                cause,
                roots,
                source: self,
            }),
        }
    }
}
impl OriginalGrammarLexerInputs {
    pub(in crate::runtime::chat::constraints) fn input(&self) -> &RegexVectorInput {
        &self.input
    }
    pub(in crate::runtime::chat::constraints) fn vocabulary(&self) -> &OriginalGrammarVocabulary {
        self.source.vocabulary()
    }
}

use llguidance::{
    api::ParserLimits,
    earley::regexvec::{
        LexemeSet, StateID,
        prepared::{
            PreparedRegexVector, PreparedRegexVectorError, PreparedRegexVectorOperationError,
        },
    },
};

#[derive(Debug, thiserror::Error)]
enum VectorCause {
    #[error("original lexical vector control geometry overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error("{0}")]
    BackingFunding(
        #[from]
        llguidance::derivre::raw::ParserAllocationPreparationError<HostMetadataFundingError>,
    ),
    #[error("{0}")]
    Vector(#[from] PreparedRegexVectorError<HostMetadataFundingError>),
    #[error(transparent)]
    VectorOperation(#[from] PreparedRegexVectorOperationError<HostMetadataFundingError>),
    #[error("{0}")]
    Lexer(#[from] PreparedLexerError<HostMetadataFundingError>),
    #[error(transparent)]
    LexerOperation(#[from] PreparedLexerOperationError<HostMetadataFundingError>),
    #[error("{0}")]
    Earley(#[from] llguidance::earley::PreparedEarleySeedError<HostMetadataFundingError>),
    #[error(transparent)]
    TokenParser(#[from] llguidance::earley::PreparedTokenParserError<HostMetadataFundingError>),
}
/// Actual initialized lexical state with original declaration/vocabulary/funding
/// custody retained after every expression, root, mask and state destination.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarLexerVector {
    vector: PreparedRegexVector,
    limits: ParserLimits,
    source: OriginalGrammarSlicer,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarLexerVectorError {
    #[source]
    cause: VectorCause,
    input: Option<RegexVectorInput>,
    source: OriginalGrammarSlicer,
}
impl OriginalGrammarLexerInputs {
    pub(in crate::runtime::chat::constraints) fn compile_vector(
        self,
        limits: &mut ParserLimits,
    ) -> Result<OriginalGrammarLexerVector, OriginalGrammarLexerVectorError> {
        let Self { input, source } = self;
        let mut input = Some(input);
        let result = (|| -> Result<PreparedRegexVector, VectorCause> {
            let frames = [
                size_of::<Self>(),
                size_of::<OriginalGrammarLexerVector>(),
                size_of::<OriginalGrammarLexerVectorError>(),
                size_of::<VectorCause>(),
                size_of::<Option<RegexVectorInput>>(),
                size_of::<HostMetadataFunding>(),
                size_of::<llguidance::derivre::raw::ParserAllocationFunding>(),
                size_of::<
                    Result<
                        llguidance::derivre::raw::ParserAllocationFunding,
                        llguidance::derivre::raw::ParserAllocationPreparationError<
                            HostMetadataFundingError,
                        >,
                    >,
                >(),
                size_of::<&mut ParserLimits>(),
                size_of::<ParserLimits>(),
                size_of::<
                    Result<PreparedRegexVector, PreparedRegexVectorError<HostMetadataFundingError>>,
                >(),
                size_of::<Result<OriginalGrammarLexerVector, OriginalGrammarLexerVectorError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
            ];
            source.vocabulary().funding.reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(VectorCause::Overflow)?,
            )?;
            let funding = source.vocabulary().funding.clone();
            let backing =
                llguidance::derivre::raw::ParserAllocationFunding::prepare(move |bytes| {
                    funding.reserve_metadata(bytes)
                })?;
            Ok(PreparedRegexVector::prepare_with_backing(
                input.take().expect("retained lexical input"),
                limits,
                Some(backing),
                &|bytes| source.vocabulary().funding.reserve_metadata(bytes),
            )?)
        })();
        match result {
            Ok(vector) => Ok(OriginalGrammarLexerVector {
                vector,
                limits: limits.clone(),
                source,
            }),
            Err(cause) => Err(OriginalGrammarLexerVectorError {
                cause,
                input,
                source,
            }),
        }
    }
}
impl OriginalGrammarLexerVector {
    pub(in crate::runtime::chat::constraints) fn vector(&self) -> &PreparedRegexVector {
        &self.vector
    }
    pub(in crate::runtime::chat::constraints) fn vocabulary(&self) -> &OriginalGrammarVocabulary {
        self.source.vocabulary()
    }
    fn operation<T, F>(&mut self, run: F) -> Result<T, OriginalGrammarLexerOperationError>
    where
        F: FnOnce(
            &mut PreparedRegexVector,
            &HostMetadataFunding,
        )
            -> Result<T, PreparedRegexVectorOperationError<HostMetadataFundingError>>,
    {
        let funding = &self.source.vocabulary().funding;
        let result = (|| -> Result<T, OperationCause> {
            let frames = [
                size_of::<OriginalGrammarLexerOperationError>(),
                size_of::<OperationCause>(),
                size_of::<HostMetadataFunding>(),
                size_of::<&mut Self>(),
                size_of::<F>(),
                size_of::<Result<T, OriginalGrammarLexerOperationError>>(),
                size_of::<Result<T, PreparedRegexVectorOperationError<HostMetadataFundingError>>>(),
                size_of::<Result<T, OperationCause>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
            ];
            funding.reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(OperationCause::Overflow)?,
            )?;
            Ok(run(&mut self.vector, funding)?)
        })();
        result.map_err(|cause| OriginalGrammarLexerOperationError {
            cause,
            funding: funding.clone(),
        })
    }
    pub(in crate::runtime::chat::constraints) fn initial_state(
        &mut self,
        selected: &LexemeSet,
    ) -> Result<StateID, OriginalGrammarLexerOperationError> {
        self.operation(|vector, funding| {
            vector.initial_state(selected, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn transition(
        &mut self,
        state: StateID,
        byte: u8,
    ) -> Result<StateID, OriginalGrammarLexerOperationError> {
        self.operation(|vector, funding| {
            vector.transition(state, byte, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn check_subsume(
        &mut self,
        state: StateID,
        lexeme: llguidance::earley::lexerspec::LexemeIdx,
        budget: u64,
    ) -> Result<bool, OriginalGrammarLexerOperationError> {
        self.operation(|vector, funding| {
            vector.check_subsume(state, lexeme, budget, &|bytes| {
                funding.reserve_metadata(bytes)
            })
        })
    }
    pub(in crate::runtime::chat::constraints) fn next_byte(
        &mut self,
        state: StateID,
    ) -> Result<llguidance::earley::regexvec::NextByte, OriginalGrammarLexerOperationError> {
        self.operation(|vector, funding| {
            vector.next_byte(state, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn possible_hidden_len(
        &mut self,
        state: StateID,
    ) -> Result<usize, OriginalGrammarLexerOperationError> {
        self.operation(|vector, funding| {
            vector.possible_lookahead_len(state, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn limit_state_to(
        &mut self,
        state: StateID,
        selected: &LexemeSet,
    ) -> Result<StateID, OriginalGrammarLexerOperationError> {
        self.operation(|vector, funding| {
            vector.limit_state_to(state, selected, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn lexeme_weight(
        &mut self,
        lexeme: llguidance::earley::lexerspec::LexemeIdx,
    ) -> Result<u32, OriginalGrammarLexerOperationError> {
        self.operation(|vector, funding| {
            vector.lexeme_weight(lexeme, &|bytes| funding.reserve_metadata(bytes))
        })
    }
}
/// An escaped operation error may own a failed mask allocation. Its actual
/// funding account remains live after the lexical owner itself is dropped.
// Mutable operations retain their actual lexer/chart in the owning caller.
// Cold constructor failures must not enlarge every token callback's transport.
#[derive(Debug, thiserror::Error)]
enum OperationCause {
    #[error("original lexical operation control geometry overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Lexer(#[from] PreparedLexerOperationError<HostMetadataFundingError>),
    #[error(transparent)]
    TokenParser(#[from] llguidance::earley::PreparedTokenParserError<HostMetadataFundingError>),
    #[error(transparent)]
    Vector(#[from] PreparedRegexVectorOperationError<HostMetadataFundingError>),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarLexerOperationError {
    #[source]
    cause: OperationCause,
    funding: HostMetadataFunding,
}

use llguidance::earley::{
    LexerResult, PreparedLexer, PreparedLexerError, PreparedLexerOperationError,
};
/// The warmed mutable lexer retires before its original declaration, vocabulary,
/// slicer and source funding. The already copied LexerSpec remains in that source.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarLexer {
    // The existing constructor and independent copy admit this cell before allocation.
    // Token operations move only its owner, never the full lexical tables.
    lexer: Box<PreparedLexer>,
    limits: ParserLimits,
    source: std::sync::Arc<OriginalGrammarSlicer>,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarLexerError {
    #[source]
    cause: VectorCause,
    vector: Option<PreparedRegexVector>,
    source: OriginalGrammarSlicer,
}
impl OriginalGrammarLexerVector {
    pub(in crate::runtime::chat::constraints) fn compile_lexer(
        self,
    ) -> Result<OriginalGrammarLexer, OriginalGrammarLexerError> {
        let Self {
            vector,
            limits,
            source,
        } = self;
        let mut vector = Some(vector);
        let result = (|| -> Result<PreparedLexer, VectorCause> {
            let parts = [
                size_of::<Self>(),
                size_of::<OriginalGrammarLexer>(),
                size_of::<PreparedLexer>(),
                size_of::<Box<PreparedLexer>>(),
                eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<OriginalGrammarSlicer>(
                )
                .ok_or(VectorCause::Overflow)?,
                size_of::<std::sync::Arc<OriginalGrammarSlicer>>(),
                size_of::<HostMetadataFunding>(),
                size_of::<OriginalGrammarLexerError>(),
                size_of::<VectorCause>(),
                size_of::<Option<PreparedRegexVector>>(),
                size_of::<Result<PreparedLexer, PreparedLexerError<HostMetadataFundingError>>>(),
                size_of::<Result<PreparedLexer, VectorCause>>(),
                size_of::<Result<OriginalGrammarLexer, OriginalGrammarLexerError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
            ];
            let funding = &source.vocabulary().funding;
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(VectorCause::Overflow)?,
            )?;
            Ok(PreparedLexer::prepare(
                vector.take().expect("retained lexical vector"),
                &|bytes| funding.reserve_metadata(bytes),
            )?)
        })();
        match result {
            Ok(lexer) => {
                let funding = source.vocabulary().funding.clone();
                Ok(OriginalGrammarLexer {
                    lexer: Box::new(lexer),
                    limits,
                    source: std::sync::Arc::new(source),
                    funding,
                })
            }
            Err(cause) => Err(OriginalGrammarLexerError {
                cause,
                vector,
                source,
            }),
        }
    }
}
impl OriginalGrammarLexer {
    pub(in crate::runtime::chat::constraints) fn lexer(&self) -> &PreparedLexer {
        &self.lexer
    }
    pub(in crate::runtime::chat::constraints) fn vocabulary(&self) -> &OriginalGrammarVocabulary {
        self.source.vocabulary()
    }
    fn operation<T, F>(&mut self, run: F) -> Result<T, OriginalGrammarLexerOperationError>
    where
        F: FnOnce(
            &mut PreparedLexer,
            &HostMetadataFunding,
            &OriginalGrammarVocabulary,
        ) -> Result<T, PreparedLexerOperationError<HostMetadataFundingError>>,
    {
        let funding = &self.funding;
        let result = (|| -> Result<T, OperationCause> {
            let parts = [
                size_of::<OriginalGrammarLexerOperationError>(),
                size_of::<OperationCause>(),
                size_of::<HostMetadataFunding>(),
                size_of::<&mut Self>(),
                size_of::<F>(),
                size_of::<Result<T, OriginalGrammarLexerOperationError>>(),
                size_of::<Result<T, PreparedLexerOperationError<HostMetadataFundingError>>>(),
                size_of::<Result<T, OperationCause>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(OperationCause::Overflow)?,
            )?;
            Ok(run(&mut self.lexer, funding, self.source.vocabulary())?)
        })();
        result.map_err(|cause| OriginalGrammarLexerOperationError {
            cause,
            funding: funding.clone(),
        })
    }
    pub(in crate::runtime::chat::constraints) fn start_state(
        &mut self,
        selected: &LexemeSet,
    ) -> Result<StateID, OriginalGrammarLexerOperationError> {
        self.operation(|lexer, funding, _| {
            lexer.start_state(selected, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn advance(
        &mut self,
        state: StateID,
        byte: u8,
    ) -> Result<LexerResult, OriginalGrammarLexerOperationError> {
        self.operation(|lexer, funding, _| {
            lexer.advance(state, byte, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn next_byte(
        &mut self,
        state: StateID,
    ) -> Result<llguidance::earley::regexvec::NextByte, OriginalGrammarLexerOperationError> {
        self.operation(|lexer, funding, _| {
            lexer.next_byte(state, &|bytes| funding.reserve_metadata(bytes))
        })
    }
    pub(in crate::runtime::chat::constraints) fn precompute_for(
        &mut self,
        selected: &LexemeSet,
    ) -> Result<(), OriginalGrammarLexerOperationError> {
        self.operation(|lexer, funding, vocabulary| {
            lexer.precompute_for(vocabulary.trie_source().trie(), selected, &|bytes| {
                funding.reserve_metadata(bytes)
            })
        })
    }
    pub(in crate::runtime::chat::constraints) fn prepare_parser_lexer(
        &mut self,
        limits: &ParserLimits,
    ) -> Result<(), OriginalGrammarLexerOperationError> {
        self.limits = self.operation(|lexer, funding, vocabulary| {
            lexer.prepare_large_lexemes(vocabulary.trie_source().trie(), limits, &|bytes| {
                funding.reserve_metadata(bytes)
            })?;
            Ok(limits.clone())
        })?;
        Ok(())
    }
}

/// Actual initial Earley chart beside the same warmed lexer/source.
/// A lexical row can be published here; this is not a runnable controller.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarEarleySeed {
    seed: llguidance::earley::PreparedEarleySeed,
    lexer: OriginalGrammarLexer,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarEarleySeedError {
    #[source]
    cause: VectorCause,
    lexer: OriginalGrammarLexer,
}
impl OriginalGrammarLexer {
    pub(in crate::runtime::chat::constraints) fn compile_earley_seed(
        self,
    ) -> Result<OriginalGrammarEarleySeed, OriginalGrammarEarleySeedError> {
        let result = (|| -> Result<llguidance::earley::PreparedEarleySeed, VectorCause> {
            let parts = [
                size_of::<Self>(),
                size_of::<OriginalGrammarEarleySeed>(),
                size_of::<OriginalGrammarEarleySeedError>(),
                size_of::<VectorCause>(),
                size_of::<llguidance::earley::SharedGrammar>(),
                size_of::<Result<llguidance::earley::PreparedEarleySeed, VectorCause>>(),
                size_of::<
                    Result<
                        llguidance::earley::PreparedEarleySeed,
                        llguidance::earley::PreparedEarleySeedError<HostMetadataFundingError>,
                    >,
                >(),
                size_of::<Result<OriginalGrammarEarleySeed, OriginalGrammarEarleySeedError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(&Self, &HostMetadataFunding)>(),
            ];
            let vocabulary = self.source.vocabulary();
            let funding = &self.funding;
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(VectorCause::Overflow)?,
            )?;
            Ok(llguidance::earley::PreparedEarleySeed::prepare(
                vocabulary.grammar_owner(),
                &|bytes| funding.reserve_metadata(bytes),
            )?
            .close_initial_agenda(&|bytes| funding.reserve_metadata(bytes))?)
        })();
        match result {
            Ok(seed) => Ok(OriginalGrammarEarleySeed { seed, lexer: self }),
            Err(cause) => Err(OriginalGrammarEarleySeedError { cause, lexer: self }),
        }
    }
}
impl OriginalGrammarEarleySeed {
    pub(in crate::runtime::chat::constraints) fn publish_initial_row(
        self,
    ) -> Result<Self, OriginalGrammarEarleySeedError> {
        let Self { seed, mut lexer } = self;
        let result = (|| -> Result<llguidance::earley::PreparedEarleySeed, VectorCause> {
            let parts = [
                size_of::<Self>(),
                size_of::<OriginalGrammarEarleySeedError>(),
                size_of::<VectorCause>(),
                size_of::<Result<llguidance::earley::PreparedEarleySeed, VectorCause>>(),
                size_of::<
                    Result<
                        llguidance::earley::PreparedEarleySeed,
                        llguidance::earley::PreparedEarleySeedError<HostMetadataFundingError>,
                    >,
                >(),
                size_of::<Result<Self, OriginalGrammarEarleySeedError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(&mut PreparedLexer, &HostMetadataFunding)>(),
            ];
            let funding = &lexer.funding;
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(VectorCause::Overflow)?,
            )?;
            Ok(seed
                .publish_initial_row(&mut lexer.lexer, &|bytes| funding.reserve_metadata(bytes))?)
        })();
        match result {
            Ok(seed) => Ok(Self { seed, lexer }),
            Err(cause) => Err(OriginalGrammarEarleySeedError { cause, lexer }),
        }
    }
    pub(in crate::runtime::chat::constraints) fn push_byte(
        self,
        byte: Option<u8>,
    ) -> Result<(Self, bool, usize), OriginalGrammarEarleySeedError> {
        let Self { seed, mut lexer } = self;
        let result = (|| -> Result<_, VectorCause> {
            let frames = [
                size_of::<Self>(),
                size_of::<OriginalGrammarEarleySeedError>(),
                size_of::<VectorCause>(),
                size_of::<Result<(Self, bool, usize), OriginalGrammarEarleySeedError>>(),
                size_of::<
                    Result<
                        (llguidance::earley::PreparedEarleySeed, bool, usize),
                        llguidance::earley::PreparedEarleySeedError<HostMetadataFundingError>,
                    >,
                >(),
                size_of::<Option<u8>>(),
                size_of::<(&mut PreparedLexer, &HostMetadataFunding)>(),
            ];
            let vocabulary = lexer.source.vocabulary();
            let funding = &lexer.funding;
            funding.reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(VectorCause::Overflow)?,
            )?;
            Ok(seed.push_byte(
                &mut lexer.lexer,
                vocabulary.trie_source().trie(),
                byte,
                &|bytes| funding.reserve_metadata(bytes),
            )?)
        })();
        match result {
            Ok((seed, accepted, backtrack)) => Ok((Self { seed, lexer }, accepted, backtrack)),
            Err(cause) => Err(OriginalGrammarEarleySeedError { cause, lexer }),
        }
    }
    pub(in crate::runtime::chat::constraints) fn scan_token_mask(
        self,
        start: &[u8],
    ) -> Result<Self, OriginalGrammarEarleySeedError> {
        let Self { seed, mut lexer } = self;
        let result = (|| -> Result<_, VectorCause> {
            let frames = [
                size_of::<Self>(),
                size_of::<OriginalGrammarEarleySeedError>(),
                size_of::<VectorCause>(),
                size_of::<Result<Self, OriginalGrammarEarleySeedError>>(),
                size_of::<
                    Result<
                        llguidance::earley::PreparedEarleySeed,
                        llguidance::earley::PreparedEarleySeedError<HostMetadataFundingError>,
                    >,
                >(),
                size_of::<Result<llguidance::earley::PreparedEarleySeed, VectorCause>>(),
                size_of::<(&mut PreparedLexer, &HostMetadataFunding, &[u8])>(),
            ];
            let vocabulary = lexer.source.vocabulary();
            let funding = &lexer.funding;
            funding.reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(VectorCause::Overflow)?,
            )?;
            Ok(seed.scan_token_mask(
                &mut lexer.lexer,
                vocabulary.trie_source().trie(),
                start,
                &|bytes| funding.reserve_metadata(bytes),
            )?)
        })();
        match result {
            Ok(seed) => Ok(Self { seed, lexer }),
            Err(cause) => Err(OriginalGrammarEarleySeedError { cause, lexer }),
        }
    }
    pub(in crate::runtime::chat::constraints) fn chop_tokens(
        self,
        tokens: &[toktrie::TokenId],
    ) -> Result<(Self, usize, usize), OriginalGrammarEarleySeedError> {
        self.token_operation(|seed, lexer, trie, _limits, funding| {
            seed.chop_tokens(lexer, trie, tokens, &|bytes| {
                funding.reserve_metadata(bytes)
            })
            .map(|(seed, count, bytes)| (seed, (count, bytes)))
        })
        .map(|(owner, (count, bytes))| (owner, count, bytes))
    }

    pub(in crate::runtime::chat::constraints) fn force_bytes(
        self,
    ) -> Result<Self, OriginalGrammarEarleySeedError> {
        self.token_operation(|seed, lexer, trie, limits, funding| {
            seed.force_bytes(lexer, trie, limits, &|bytes| {
                funding.reserve_metadata(bytes)
            })
            .map(|seed| (seed, ()))
        })
        .map(|(owner, ())| owner)
    }
    pub(in crate::runtime::chat::constraints) fn compute_token_mask(
        self,
        start: &[u8],
    ) -> Result<Self, OriginalGrammarEarleySeedError> {
        self.token_operation(|seed, lexer, trie, limits, funding| {
            seed.compute_token_mask(lexer, trie, limits, start, &|bytes| {
                funding.reserve_metadata(bytes)
            })
            .map(|seed| (seed, ()))
        })
        .map(|(owner, ())| owner)
    }
    fn token_operation<T, F>(self, run: F) -> Result<(Self, T), OriginalGrammarEarleySeedError>
    where
        F: FnOnce(
            llguidance::earley::PreparedEarleySeed,
            &mut PreparedLexer,
            &toktrie::TokTrie,
            &ParserLimits,
            &HostMetadataFunding,
        ) -> Result<
            (llguidance::earley::PreparedEarleySeed, T),
            llguidance::earley::PreparedEarleySeedError<HostMetadataFundingError>,
        >,
    {
        let Self { seed, mut lexer } = self;
        let result = (|| -> Result<_, VectorCause> {
            let frames = [
                size_of::<Self>(),
                size_of::<OriginalGrammarEarleySeedError>(),
                size_of::<VectorCause>(),
                size_of::<F>(),
                size_of::<T>(),
                size_of::<Result<(Self, T), OriginalGrammarEarleySeedError>>(),
                size_of::<
                    Result<
                        (llguidance::earley::PreparedEarleySeed, T),
                        llguidance::earley::PreparedEarleySeedError<HostMetadataFundingError>,
                    >,
                >(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(
                    &mut PreparedLexer,
                    &toktrie::TokTrie,
                    &ParserLimits,
                    &HostMetadataFunding,
                )>(),
            ];
            let vocabulary = lexer.source.vocabulary();
            let funding = &lexer.funding;
            funding.reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(VectorCause::Overflow)?,
            )?;
            Ok(run(
                seed,
                &mut lexer.lexer,
                vocabulary.trie_source().trie(),
                &lexer.limits,
                funding,
            )?)
        })();
        match result {
            Ok((seed, value)) => Ok((Self { seed, lexer }, value)),
            Err(cause) => Err(OriginalGrammarEarleySeedError { cause, lexer }),
        }
    }
    pub(in crate::runtime::chat::constraints) fn apply_token(
        self,
        bytes: &[u8],
        token: toktrie::TokenId,
    ) -> Result<(Self, usize), OriginalGrammarEarleySeedError> {
        self.token_operation(|seed, lexer, trie, limits, funding| {
            seed.apply_token(lexer, trie, limits, bytes, token, &|bytes| {
                funding.reserve_metadata(bytes)
            })
        })
    }
    pub(in crate::runtime::chat::constraints) fn validate_tokens(
        self,
        tokens: &[toktrie::TokenId],
    ) -> Result<(Self, usize), OriginalGrammarEarleySeedError> {
        self.token_operation(|seed, lexer, trie, limits, reserve| {
            seed.validate_tokens(lexer, trie, limits, tokens, &|bytes| {
                reserve.reserve_metadata(bytes)
            })
        })
    }

    pub(in crate::runtime::chat::constraints) fn scan_eos(
        self,
    ) -> Result<(Self, bool), OriginalGrammarEarleySeedError> {
        self.token_operation(|seed, lexer, trie, limits, funding| {
            seed.scan_eos(lexer, trie, limits, &|bytes| {
                funding.reserve_metadata(bytes)
            })
        })
    }
    pub(in crate::runtime::chat::constraints) fn is_accepting(
        self,
    ) -> Result<(Self, bool), OriginalGrammarEarleySeedError> {
        self.token_operation(|seed, lexer, trie, limits, funding| {
            seed.is_accepting(lexer, trie, limits, &|bytes| {
                funding.reserve_metadata(bytes)
            })
        })
    }
    pub(in crate::runtime::chat::constraints) fn seed(
        &self,
    ) -> &llguidance::earley::PreparedEarleySeed {
        &self.seed
    }
    pub(in crate::runtime::chat::constraints) fn vocabulary(&self) -> &OriginalGrammarVocabulary {
        self.lexer.vocabulary()
    }
}

/// Actual token-level mutable owner; all chart/history arrays retire before the
/// retained lexer/vocabulary/declaration/funding chain.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenParser {
    parser: llguidance::earley::PreparedTokenParser,
    lexer: OriginalGrammarLexer,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenParserError {
    #[source]
    cause: OperationCause,
    unstarted: Option<llguidance::earley::PreparedEarleySeed>,
    pending: Option<llguidance::earley::PreparedTokenParser>,
    lexer: OriginalGrammarLexer,
}
impl OriginalGrammarEarleySeed {
    pub(in crate::runtime::chat::constraints) fn into_token_parser(
        self,
    ) -> Result<OriginalGrammarTokenParser, OriginalGrammarTokenParserError> {
        let Self { seed, mut lexer } = self;
        let mut unstarted = Some(seed);
        let result = (|| -> Result<_, OperationCause> {
            let parts = [
                size_of::<Self>(),
                size_of::<OriginalGrammarTokenParser>(),
                size_of::<OriginalGrammarTokenParserError>(),
                size_of::<OperationCause>(),
                size_of::<Option<llguidance::earley::PreparedEarleySeed>>(),
                size_of::<Result<llguidance::earley::PreparedTokenParser, OperationCause>>(),
                size_of::<Result<OriginalGrammarTokenParser, OriginalGrammarTokenParserError>>(),
                size_of::<
                    Result<
                        llguidance::earley::PreparedTokenParser,
                        llguidance::earley::PreparedTokenParserError<HostMetadataFundingError>,
                    >,
                >(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(
                    &mut PreparedLexer,
                    &toktrie::TokTrie,
                    &ParserLimits,
                    Option<usize>,
                    &HostMetadataFunding,
                )>(),
            ];
            let vocabulary = lexer.source.vocabulary();
            let funding = &lexer.funding;
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(OperationCause::Overflow)?,
            )?;
            Ok(llguidance::earley::PreparedTokenParser::prepare(
                unstarted.take().expect("token session source"),
                &mut lexer.lexer,
                vocabulary.trie_source().trie(),
                &lexer.limits,
                vocabulary.recipe.grammar_max_tokens(),
                &|bytes| funding.reserve_metadata(bytes),
            )?)
        })();
        match result {
            Ok(parser) => Ok(OriginalGrammarTokenParser { parser, lexer }),
            Err(cause) => Err(OriginalGrammarTokenParserError {
                cause,
                unstarted,
                pending: None,
                lexer,
            }),
        }
    }
}
impl OriginalGrammarTokenParser {
    fn operation<T, F>(self, run: F) -> Result<(Self, T), OriginalGrammarTokenParserError>
    where
        F: FnOnce(
            llguidance::earley::PreparedTokenParser,
            &mut PreparedLexer,
            &toktrie::TokTrie,
            &ParserLimits,
            &HostMetadataFunding,
        ) -> Result<
            (llguidance::earley::PreparedTokenParser, T),
            llguidance::earley::PreparedTokenParserError<HostMetadataFundingError>,
        >,
    {
        let Self { parser, mut lexer } = self;
        let mut pending = Some(parser);
        let result = (|| -> Result<_, OperationCause> {
            let parts = [
                size_of::<Self>(),
                size_of::<OriginalGrammarTokenParserError>(),
                size_of::<OperationCause>(),
                size_of::<F>(),
                size_of::<T>(),
                size_of::<Option<llguidance::earley::PreparedTokenParser>>(),
                size_of::<Result<(Self, T), OriginalGrammarTokenParserError>>(),
                size_of::<
                    Result<
                        (llguidance::earley::PreparedTokenParser, T),
                        llguidance::earley::PreparedTokenParserError<HostMetadataFundingError>,
                    >,
                >(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(
                    &mut PreparedLexer,
                    &toktrie::TokTrie,
                    &ParserLimits,
                    &HostMetadataFunding,
                )>(),
            ];
            let vocabulary = lexer.source.vocabulary();
            let funding = &lexer.funding;
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(OperationCause::Overflow)?,
            )?;
            Ok(run(
                pending.take().expect("token session owner"),
                &mut lexer.lexer,
                vocabulary.trie_source().trie(),
                &lexer.limits,
                funding,
            )?)
        })();
        match result {
            Ok((parser, value)) => Ok((Self { parser, lexer }, value)),
            Err(cause) => Err(OriginalGrammarTokenParserError {
                cause,
                unstarted: None,
                pending,
                lexer,
            }),
        }
    }
    pub(in crate::runtime::chat::constraints) fn try_consume_tokens(
        self,
        tokens: &[toktrie::TokenId],
    ) -> Result<(Self, usize), OriginalGrammarTokenParserError> {
        self.operation(|parser, lexer, trie, limits, funding| {
            parser.try_consume_tokens(lexer, trie, limits, tokens, &|bytes| {
                funding.reserve_metadata(bytes)
            })
        })
    }
    pub(in crate::runtime::chat::constraints) fn is_accepting(
        self,
    ) -> Result<(Self, bool), OriginalGrammarTokenParserError> {
        self.operation(|parser, lexer, trie, limits, funding| {
            parser.is_accepting(lexer, trie, limits, &|bytes| {
                funding.reserve_metadata(bytes)
            })
        })
    }
    pub(in crate::runtime::chat::constraints) fn rollback(
        self,
        tokens: usize,
    ) -> Result<Self, OriginalGrammarTokenParserError> {
        self.operation(|parser, lexer, trie, limits, funding| {
            parser
                .rollback(lexer, trie, limits, tokens, &|bytes| {
                    funding.reserve_metadata(bytes)
                })
                .map(|parser| (parser, ()))
        })
        .map(|(owner, ())| owner)
    }
    pub(in crate::runtime::chat::constraints) fn reset(
        self,
    ) -> Result<Self, OriginalGrammarTokenParserError> {
        let tokens = self.parser.tokens().len();
        self.rollback(tokens)
    }
    pub(in crate::runtime::chat::constraints) fn parser(
        &self,
    ) -> &llguidance::earley::PreparedTokenParser {
        &self.parser
    }
    pub(in crate::runtime::chat::constraints) fn vocabulary(
        &self,
    ) -> &super::OriginalGrammarVocabulary {
        self.lexer.source.vocabulary()
    }
}
