pub use derivre::prepared_funding::{PreparedFunding, Frame, FrameError};
mod from_guidance;
pub(crate) use from_guidance::is_grammar_storage_failure;
mod grammar;
pub(crate) mod lexer;
mod parser;
mod slicer;

pub mod lexerspec;
pub mod perf;
pub mod regexvec;

pub use from_guidance::{GrammarCompilationError, ValidationResult};
#[allow(unused_imports)]
pub use grammar::{
    BitIdx, CGrammar, CSymIdx, Grammar, ParamCond, ParamExpr, ParamRef, ParamValue, SymIdx,
    SymbolProps, SharedGrammar, SharedGrammarFailure,
};
pub use parser::{BiasComputer, Parser, ParserError, ParserMetrics, ParserRecognizer, ParserStats};
pub use slicer::source::{
    SlicerSource, SlicerRecognitionError, SlicerSourceDescriptor, SlicerSourceError, SlicerSourceFailure, SlicerSourcePlan,
    SlicerSourceRequirements, SlicerSourceView,
};
pub use slicer::SlicedBiasComputer;

pub use slicer::source::prepared::{
    SlicerConstructionFailure, SlicerConstructionPlan, SlicerConstructionRequirements,
    SlicerProgram,
};

pub use slicer::step::{SlicerStep, SlicerStepFailure, SlicerStepPlan, SlicerStepRequirements};

pub use lexer::{
    LexerPrecompute, LexerPrecomputeFailure, LexerPrecomputePlan, LexerPrecomputeRequirements,
};

pub use grammar::{
    CompiledGrammarCopyFailure, CompiledGrammarCopyPlan, CompiledGrammarCopyRequirements, ConditionSourceError,
};

// Narrow prepared lexical handoff for the facade-owned controller.
pub use lexer::prepared::{PreparedLexer, PreparedLexerCopyFailure, PreparedLexerError, PreparedLexerOperationError};
pub use lexer::{Lexer, LexerResult};

pub use parser::{
    PreparedEarleySeed, PreparedEarleySeedError, PreparedTokenParser, PreparedTokenParserError,
};
