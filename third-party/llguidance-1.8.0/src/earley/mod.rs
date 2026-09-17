mod from_guidance;
mod grammar;
pub(crate) mod lexer;
mod parser;
mod slicer;

pub mod lexerspec;
pub mod perf;
pub mod regexvec;

pub use from_guidance::ValidationResult;
#[allow(unused_imports)]
pub use grammar::{
    BitIdx, CGrammar, CSymIdx, Grammar, ParamCond, ParamExpr, ParamRef, ParamValue, SymIdx,
    SymbolProps,
};
pub use parser::{BiasComputer, Parser, ParserError, ParserMetrics, ParserRecognizer, ParserStats};
pub use slicer::source::{
    SlicerSource, SlicerSourceDescriptor, SlicerSourceError, SlicerSourceFailure, SlicerSourcePlan,
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
    CompiledGrammarCopyFailure, CompiledGrammarCopyPlan, CompiledGrammarCopyRequirements,
};

// Narrow prepared lexical handoff for the facade-owned controller.
pub use lexer::prepared::{PreparedLexer, PreparedLexerCopyFailure, PreparedLexerError, PreparedLexerOperationError};
pub use lexer::{Lexer, LexerResult};

pub use parser::{
    PreparedEarleySeed, PreparedEarleySeedError, PreparedTokenParser, PreparedTokenParserError,
};
