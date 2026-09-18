use std::alloc::Layout;
use std::collections::TryReserveError;
use std::fmt;
use std::mem::size_of;

use crate::{AutoEscape, UndefinedBehavior, default_auto_escape_callback};

use super::recipe;

/// Exact decoded source of the checked compiler image. This readonly static
/// text is profile data, never an allocation grant or source-owner witness.
/// Adapters for encoded input must retain and validate their genuine input
/// borrow before using this equivalent decoded value for fresh construction.
pub fn supported_source() -> &'static str {
    recipe::SOURCE
}

/// Actual template name selected by the caller's source policy.
#[derive(Clone, Copy, Debug)]
pub enum TemplateName<'a> {
    /// An unqualified template uses this exact existing name.
    Single(&'a str),
    /// A selected checkpoint entry uses the ordinary composed name.
    Named {
        /// Existing model identifier, borrowed through compilation.
        model: &'a str,
        /// Exact checkpoint entry selected by the request.
        entry: &'a str,
    },
}
impl<'a> TemplateName<'a> {
    pub(super) fn parts(self) -> [&'a str; 3] {
        match self {
            Self::Single(name) => [name, "", ""],
            Self::Named { model, entry } => [model, "::chat_template::", entry],
        }
    }
    pub(super) fn byte_len(self) -> Result<usize, SourceError> {
        let parts = self.parts();
        parts[0]
            .len()
            .checked_add(parts[1].len())
            .and_then(|n| n.checked_add(parts[2].len()))
            .ok_or(SourceError::Overflow)
    }
    pub(super) fn escape(self) -> AutoEscape {
        match self {
            Self::Single(name) => default_auto_escape_callback(name),
            Self::Named { entry, .. } => crate::defaults::auto_escape_from_name(entry, true),
        }
    }
}

/// The actual scalar ordinary settings represented by the emitted source.
#[derive(Clone, Copy, Debug)]
pub struct TemplateSettings {
    /// Strip the first newline after a block.
    pub trim_blocks: bool,
    /// Strip leading block whitespace.
    pub lstrip_blocks: bool,
    /// Retain a trailing template newline.
    pub keep_trailing_newline: bool,
    /// Ordinary undefined-value policy.
    pub undefined_behavior: UndefinedBehavior,
}
impl TemplateSettings {
    /// The settings installed by the ordinary text chat environment. The closed
    /// source profile uses its registered Python-compatible tojson spelling,
    /// not the vendor HTML-escaping filter or an arbitrary replacement callback.
    pub const fn text_chat() -> Self {
        Self {
            trim_blocks: true,
            lstrip_blocks: true,
            keep_trailing_newline: false,
            undefined_behavior: UndefinedBehavior::Lenient,
        }
    }
    pub(super) fn matches(self) -> bool {
        self.trim_blocks
            && self.lstrip_blocks
            && !self.keep_trailing_newline
            && self.undefined_behavior == UndefinedBehavior::Lenient
    }
}

/// Fixed source rejection, requiring no allocated error string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceError {
    /// The source uses an operation outside the checked construction worker.
    Profile,
    /// Actual compilation settings or selected escaping differ.
    Settings,
    /// Checked destination arithmetic exceeded the addressable layout.
    Overflow,
    /// The checked source or image did not satisfy its storage/control-flow invariants.
    Geometry,
    /// The shared parser rejected the actual source; detailed borrowed syntax
    /// storage retires under the same compiler allowance before this escapes.
    Syntax,
}
impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Profile => "chat source construction profile is unfinished",
            Self::Settings => "chat source settings differ from the compiled image",
            Self::Overflow => "chat source layout overflow",
            Self::Geometry => "invalid checked chat instruction geometry",
            Self::Syntax => "invalid chat template syntax",
        })
    }
}
impl std::error::Error for SourceError {}

/// The three actual source destinations, in construction order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceBuffer {
    /// Pointer-free ordinary instruction records.
    Instructions,
    /// Copied source, literal/lookup strings and actual selected name.
    Bytes,
    /// One actual source location per emitted instruction.
    Locations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Range {
    pub start: usize,
    pub len: usize,
}
impl Range {
    pub const fn new(start: usize, len: usize) -> Self {
        Self { start, len }
    }
    pub fn text(self, bytes: &[u8]) -> Option<&str> {
        let end = self.start.checked_add(self.len)?;
        std::str::from_utf8(bytes.get(self.start..end)?).ok()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MappingProjection {
    Keys,
    Values,
    Items,
}
impl MappingProjection {
    pub(super) fn method(self) -> &'static str {
        match self {
            Self::Keys => "keys",
            Self::Values => "values",
            Self::Items => "items",
        }
    }
}

// Actual source order of at most four registered filter options. A literal
// separators pair contributes two operand values without allocating a list.
#[derive(Clone, Copy, Debug)]
pub(super) struct JsonOptions {
    pub order: [u8; 4],
    pub arguments: u8,
    pub separator_pair: bool,
}
impl JsonOptions {
    pub const NAMES: [&'static str;4]=["ensure_ascii","indent","separators","sort_keys"];
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Instruction {
    Lookup(Range),
    StoreLocal(Range),
    SetAttr(Range),
    BuildKwargs(usize),
    CallFunction(Range, u16),
    DupTop,
    DiscardTop,
    Argument(Range),
    Arguments { start: u32, count: usize },
    Enclose(Range),
    GetClosure,
    BuildMacro(Range, u32, u8),
    Return,
    UnpackList(usize),
    BuildList(usize),
    BeginCollection,
    AppendCollection,
    EndCollection,
    BuildMap(usize),
    GetAttr(Range),
    GetItem,
    Slice,
    MappingGet(u8),
    MappingView(MappingProjection),
    Items,
    List,
    Selection { invert:bool, attribute:bool, args:u8 },
    StartsWith,
    EndsWith,
    StringSplit(u8),
    StringStrip { args: u8, left: bool, right: bool },
    Literal(Range),
    Undefined,
    Zero,
    Scalar(crate::value::primitive::scalar::Scalar),
    Add,
    StringConcat,
    Arithmetic(crate::value::primitive::scalar::Arithmetic),
    Ne,
    Eq,
    Order(crate::vm::shared::Order),
    In,
    Not,
    Neg,
    Presence(crate::value::primitive::Presence),
    TypeTest(crate::value::primitive::type_tests::TypeTest),
    Trim,
    TrimCharacters,
    Upper,
    MapFilter,
    DictSort,
    Length,
    Stringify,
    Floatify,
    ToJson(JsonOptions),
    Join(u8),
    Replace,
    StringReplace,
    Default(u8),
    EmptySequence,
    BeginCapture,
    EndCapture,
    Emit,
    PushLoop(u8),
    Iterate(u32),
    PopLoopFrame,
    Jump(u32),
    JumpIfFalse(u32),
    JumpIfFalseOrPop(u32),
    JumpIfTrueOrPop(u32),
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub start_offset: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub end_offset: usize,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Location {
    pub line: usize,
    pub span: Option<Span>,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Geometry {
    pub operands: usize,
    pub frames: usize,
    pub locals: usize,
}

/// Checked requested layouts; no public constructor accepts a byte grant.
#[derive(Clone, Copy, Debug)]
pub struct CompileRequirements {
    pub(super) capacities: [usize; 3],
    pub(super) buffers: usize,
    pub(super) required: usize,
}
impl CompileRequirements {
    /// Requested element count of one actual destination.
    pub fn capacity(self, buffer: SourceBuffer) -> usize {
        self.capacities[index(buffer)]
    }
    /// Sum of requested compiler and output allocation layouts.
    pub fn buffer_bytes(self) -> usize {
        self.buffers
    }
    /// Buffer layouts and concrete plan/partial/return controls, excluding a
    /// caller's future Arc/account wrapper, which it must price separately.
    pub fn required_bytes(self) -> usize {
        self.required
    }
}
fn index(buffer: SourceBuffer) -> usize {
    match buffer {
        SourceBuffer::Instructions => 0,
        SourceBuffer::Bytes => 1,
        SourceBuffer::Locations => 2,
    }
}
fn layout<T>(n: usize) -> Result<usize, SourceError> {
    Layout::array::<T>(n)
        .map(|l| l.size())
        .map_err(|_| SourceError::Overflow)
}

/// A genuine borrowed source and name, retained until the one construction ends.
/// It is neither Clone nor an owning compiled-template adoption API.
#[derive(Debug)]
pub struct CompilePlan<'a> {
    source: &'a str,
    name: TemplateName<'a>,
    geometry: Geometry,
    requirements: CompileRequirements,
    #[cfg(feature = "development-closed-chat")]
    fail: Option<SourceBuffer>,
}
impl<'a> CompilePlan<'a> {
    /// Select an image by exact decoded source and actual scalar settings/name.
    /// No Environment, parser, compiler cache or allocation is constructed.
    pub fn prepare(
        source: &'a str,
        name: TemplateName<'a>,
        settings: TemplateSettings,
    ) -> Result<Self, SourceError> {
        if source != recipe::SOURCE {
            return Err(SourceError::Profile);
        }
        if !settings.matches() || name.escape() != AutoEscape::None {
            return Err(SourceError::Settings);
        }
        let geometry = validate_image()?;
        let bytes = recipe::BYTES
            .len()
            .checked_add(name.byte_len()?)
            .ok_or(SourceError::Overflow)?;
        let capacities = [recipe::INSTRUCTIONS.len(), bytes, recipe::LOCATIONS.len()];
        let buffers = layout::<Instruction>(capacities[0])?
            .checked_add(layout::<u8>(capacities[1])?)
            .and_then(|n| n.checked_add(layout::<Location>(capacities[2]).ok()?))
            .ok_or(SourceError::Overflow)?;
        // Explicit fixed validation/constructor populations accompany the
        // requested Vec layouts. This sum names Rust control representations;
        // it is not a claim about generated machine-stack frame sizes.
        let controls = [
            size_of::<Self>(),
            size_of::<PreparedTemplate>(),
            size_of::<Result<PreparedTemplate, CompileError>>(),
            size_of::<CompileError>(),
            size_of::<CompileCause>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<CompileRequirements>(),
            size_of::<Geometry>(),
            size_of::<Result<Geometry, SourceError>>(),
            size_of::<[Option<usize>; 40]>(), // actual validate_image worklist
            size_of::<[(usize, usize); 2]>(), // its two actual outgoing edges
            size_of::<(usize, isize)>(),      // minimum/delta of the current opcode
            size_of::<[usize; 6]>(),          // pc/depth/after/maximum/target/value
            size_of::<bool>(),                // changed flag
            size_of::<[usize; 3]>(),          // requested capacities
            size_of::<[&str; 2]>(),           // selected name parts
            size_of::<Layout>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(SourceError::Overflow)?;
        let required = buffers.checked_add(controls).ok_or(SourceError::Overflow)?;
        Ok(Self {
            source,
            name,
            geometry,
            requirements: CompileRequirements {
                capacities,
                buffers,
                required,
            },
            #[cfg(feature = "development-closed-chat")]
            fail: None,
        })
    }
    /// Exact requirements before any construction reserve.
    pub fn requirements(&self) -> CompileRequirements {
        self.requirements
    }
    /// Development-only capacity overflow at the selected real reserve site.
    #[cfg(feature = "development-closed-chat")]
    pub fn fail_reservation(mut self, buffer: SourceBuffer) -> Self {
        self.fail = Some(buffer);
        self
    }
    fn requested(&self, buffer: SourceBuffer) -> usize {
        #[cfg(feature = "development-closed-chat")]
        if self.fail == Some(buffer) {
            return usize::MAX;
        }
        self.requirements.capacity(buffer)
    }
    /// Attempt each real allocation once, retaining all completed prefixes on
    /// failure. No shrink, recompile, retry, or accounting handoff follows.
    pub fn compile(self) -> Result<PreparedTemplate, CompileError> {
        let mut data = PreparedTemplate {
            instructions: Vec::new(),
            decoded_source: None,
            bytes: Vec::new(),
            locations: Vec::new(),
            name: Range::new(
                recipe::BYTES.len(),
                self.name.byte_len().expect("checked name"),
            ),
            geometry: self.geometry,
        };
        macro_rules! reserve {
            ($field:ident, $buffer:ident) => {
                if let Err(error) = data
                    .$field
                    .try_reserve_exact(self.requested(SourceBuffer::$buffer))
                {
                    return Err(CompileError {
                        data,
                        cause: CompileCause::Reserve(SourceBuffer::$buffer, error),
                    });
                }
            };
        }
        reserve!(instructions, Instructions);
        data.instructions.extend_from_slice(&recipe::INSTRUCTIONS);
        reserve!(bytes, Bytes);
        // Copy the actual borrow, then the already source-checked decoded image
        // strings. Runtime addresses never enter instructions or identity.
        data.bytes.extend_from_slice(self.source.as_bytes());
        data.bytes
            .extend_from_slice(&recipe::BYTES.as_bytes()[self.source.len()..]);
        for part in self.name.parts() {
            data.bytes.extend_from_slice(part.as_bytes());
        }
        reserve!(locations, Locations);
        data.locations.extend_from_slice(&recipe::LOCATIONS);
        Ok(data)
    }
}

/// Fresh closed source storage, with readonly projections and no ordinary
/// Environment/Value/compiled-template conversion, Clone or serde escape.
#[derive(Debug)]
pub struct PreparedTemplate {
    pub(super) instructions: Vec<Instruction>,
    pub(super) bytes: Vec<u8>,
    pub(super) locations: Vec<Location>,
    pub(super) name: Range,
    pub(super) decoded_source: Option<Vec<u8>>,
    pub(super) geometry: Geometry,
}
impl PreparedTemplate {
    /// Actual selected template name retained in source storage.
    pub fn name(&self) -> &str {
        self.name.text(&self.bytes).expect("constructed name")
    }
    /// Actual decoded, source-matched template text.
    pub fn source(&self) -> &str {
        match &self.decoded_source {
            Some(bytes) => std::str::from_utf8(bytes).expect("validated source"),
            None => Range::new(0, recipe::SOURCE.len())
                .text(&self.bytes)
                .expect("constructed source"),
        }
    }
    /// Number of emitted ordinary instructions.
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }
    /// Actual retained destination capacities, also meaningful for partial errors.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.instructions.capacity() * size_of::<Instruction>()
            + self.bytes.capacity()
            + self.locations.capacity() * size_of::<Location>()
            + self.decoded_source.as_ref().map_or(0, Vec::capacity)
    }
}
/// Actual reserve failure from its selected destination.
#[derive(Debug)]
pub enum CompileCause {
    /// Source buffer and actual TryReserveError.
    Reserve(SourceBuffer, TryReserveError),
    /// Original shared-parser reserve failure. Its temporary AST buffers have
    /// already retired while the consumer's compiler account remains live.
    SyntaxReserve(TryReserveError),
    /// Fixed source/codegen qualification or syntax rejection.
    Source(SourceError),
}
impl fmt::Display for CompileCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(cause) => cause.fmt(f),
            _ => f.write_str("chat source destination reserve failed"),
        }
    }
}
impl std::error::Error for CompileCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(_, e) | Self::SyntaxReserve(e) => Some(e),
            Self::Source(cause) => Some(cause),
        }
    }
}
/// Owns the exact real partial buffers until its final destruction.
#[derive(Debug)]
pub struct CompileError {
    pub(super) data: PreparedTemplate,
    pub(super) cause: CompileCause,
}
impl CompileError {
    /// The real reserve cause, without allocating erasure.
    pub fn cause(&self) -> &CompileCause {
        &self.cause
    }
    /// Actual partial destination capacity still retained by this error.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.data.retained_buffer_bytes()
    }
}
impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

fn validate_image() -> Result<Geometry, SourceError> {
    let code = &recipe::INSTRUCTIONS;
    if code.len() != recipe::LOCATIONS.len() {
        return Err(SourceError::Geometry);
    }
    let mut depths = [None; 40];
    depths[0] = Some(0usize);
    let mut maximum = 0;
    // Each newly reachable entry is installed once; no heap worklist exists.
    for _ in 0..=code.len() {
        let mut changed = false;
        for (pc, instruction) in code.iter().enumerate() {
            let Some(depth) = depths[pc] else { continue };
            maximum = maximum.max(depth);
            let (minimum, delta) = match instruction {
                Instruction::Lookup(_)
                | Instruction::Literal(_)
                | Instruction::Undefined
                | Instruction::Zero
                | Instruction::Scalar(_)
                | Instruction::EmptySequence
                | Instruction::EndCapture
                | Instruction::EndCollection => (0, 1isize),
                Instruction::BeginCapture => (0,0),
                Instruction::BeginCollection => (1,0),
                Instruction::SetAttr(_) => (2, -2),
                Instruction::BuildKwargs(count) => {
                    let count = count.checked_mul(2).ok_or(SourceError::Overflow)?;
                    (
                        count,
                        1 - isize::try_from(count).map_err(|_| SourceError::Overflow)?,
                    )
                }
                Instruction::CallFunction(_, count) => (
                    usize::from(*count),
                    1 - isize::try_from(*count).map_err(|_| SourceError::Overflow)?,
                ),
                Instruction::StoreLocal(_)
                | Instruction::Emit
                | Instruction::AppendCollection
                | Instruction::PushLoop(_)
                | Instruction::JumpIfFalse(_) => (1, -1),
                Instruction::GetAttr(_)
                | Instruction::Not
                | Instruction::Neg
                | Instruction::Presence(_)
                | Instruction::TypeTest(_)
                | Instruction::Trim
                | Instruction::Upper
                | Instruction::DictSort
                | Instruction::Length
                | Instruction::Stringify
                | Instruction::Floatify
                | Instruction::MappingView(_)
                | Instruction::Items
                | Instruction::List
                | Instruction::JumpIfFalseOrPop(_)
                | Instruction::JumpIfTrueOrPop(_) => (1, 0),
                Instruction::BuildMap(count) => { let n=count.checked_mul(2).ok_or(SourceError::Overflow)?;(n,1isize.checked_sub(isize::try_from(n).map_err(|_|SourceError::Overflow)?).ok_or(SourceError::Overflow)?) },
                Instruction::BuildList(count) => (*count, 1isize.checked_sub(isize::try_from(*count).map_err(|_|SourceError::Overflow)?).ok_or(SourceError::Overflow)?),
                Instruction::GetItem
                | Instruction::Add
                | Instruction::StringConcat
                | Instruction::Arithmetic(_)
                | Instruction::Ne
                | Instruction::Eq
                | Instruction::In
                | Instruction::StartsWith
                | Instruction::EndsWith
                | Instruction::Order(_)
                | Instruction::TrimCharacters
                | Instruction::MapFilter => (2, -1),
                Instruction::Replace | Instruction::StringReplace => (3, -2),
                Instruction::Slice => (4, -3),
                Instruction::ToJson(options) => (usize::from(options.arguments) + 1, -(isize::from(options.arguments))),
                Instruction::Selection{args,..} if *args<=3 => (usize::from(*args)+1,-isize::from(*args)),
                Instruction::Selection{..} => return Err(SourceError::Geometry),
                Instruction::Join(args) if *args <= 1 => {
                    (usize::from(*args) + 1, -isize::from(*args))
                }
                Instruction::Join(_) => return Err(SourceError::Geometry),
                Instruction::StringSplit(args) if *args<=2 => (usize::from(*args)+1,-isize::from(*args)),
                Instruction::StringStrip { args, .. }
                    if *args <= 1 =>
                {
                    (usize::from(*args) + 1, -isize::from(*args))
                }
                Instruction::StringSplit(_) | Instruction::StringStrip { .. } => {
                    return Err(SourceError::Geometry);
                }
                Instruction::MappingGet(args) if (1..=2).contains(args) => {
                    (usize::from(*args) + 1, -isize::from(*args))
                }
                Instruction::MappingGet(_) => return Err(SourceError::Geometry),
                Instruction::Default(args) if *args <= 2 => {
                    (usize::from(*args) + 1, -isize::from(*args))
                }
                Instruction::Default(_) => return Err(SourceError::Geometry),
                _ => (0, 0),
            };
            if depth < minimum {
                return Err(SourceError::Geometry);
            }
            let after = depth
                .checked_add_signed(delta)
                .ok_or(SourceError::Geometry)?;
            let mut edges = [(pc + 1, after), (usize::MAX, 0)];
            match instruction {
                Instruction::Jump(target) => edges[0] = (*target as usize, depth),
                Instruction::Iterate(target) => {
                    edges = [(pc + 1, depth + 1), (*target as usize, depth)]
                }
                Instruction::JumpIfFalse(target) => edges[1] = (*target as usize, after),
                Instruction::JumpIfFalseOrPop(target) | Instruction::JumpIfTrueOrPop(target) => {
                    edges = [(pc + 1, depth - 1), (*target as usize, depth)]
                }
                _ => {}
            }
            for (target, value) in edges {
                if target == usize::MAX {
                    continue;
                }
                if target > code.len() || (target <= pc && (pc != 33 || target != 2)) {
                    return Err(SourceError::Geometry);
                }
                if let Some(existing) = depths[target] {
                    if existing != value {
                        return Err(SourceError::Geometry);
                    }
                } else {
                    depths[target] = Some(value);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if depths.iter().any(Option::is_none) || depths[39] != Some(0) || maximum != 3 {
        return Err(SourceError::Geometry);
    }
    if !matches!(code[1], Instruction::PushLoop(1))
        || !matches!(code[2], Instruction::Iterate(34))
        || !matches!(code[3], Instruction::StoreLocal(_))
        || !matches!(code[34], Instruction::PopLoopFrame)
    {
        return Err(SourceError::Geometry);
    }
    // These slots follow the one actual emitted message-loop frame/local; the
    // root context lives inline in the closed execution control.
    Ok(Geometry {
        operands: maximum,
        frames: 1,
        locals: 1,
    })
}
