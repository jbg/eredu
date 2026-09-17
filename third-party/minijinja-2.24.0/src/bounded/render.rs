use std::alloc::Layout;
use std::collections::TryReserveError;
use std::fmt;
use std::mem::{size_of, size_of_val};

use super::input::{Messages, RenderContext};
use super::source::{PreparedTemplate, Range};
use super::worker::macros::{CallFrame, Layout as MacroLayout};
use super::worker::namespace::{Layout as NamespaceLayout, Namespace, NamespaceField};
use super::worker::{self, Atom, Borrowed, Counts, Frame, Local, Slot, Workspace};

/// Requested temporary JSON continuation/key destinations for one measured
/// attempt. These are layout facts; they confer no allocation permission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JsonCapacity {
    /// Maximum simultaneously live nonempty container continuations.
    pub frames: usize,
    /// Borrowed entries of simultaneously live unsorted objects.
    pub keys: usize,
}

// Grow between completed attempts only. Each capacity remains a layout fact
// which the caller must admit before allocating the next fixed destination.
fn grown_capacity(current: usize, required: usize) -> usize {
    if required > current {
        required.max(current.checked_mul(2).unwrap_or(required))
    } else {
        current
    }
}
impl JsonCapacity {
    /// Preserve all destinations actually reached by completed attempts.
    pub fn union(self, other: Self) -> Self {
        Self {
            frames: self.frames.max(other.frames),
            keys: self.keys.max(other.keys),
        }
    }
    /// Amortized next-attempt layout; conveys no allocation permission.
    pub fn grown_for(self, required: Self) -> Self {
        Self {
            frames: grown_capacity(self.frames, required.frames),
            keys: grown_capacity(self.keys, required.keys),
        }
    }
}

/// Actual generated-text prefix required by a reached text consumer. These
/// capacities describe temporary storage, never allocation authority.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextCapacity {
    /// Existing concatenation atom records required by the prefix.
    pub atoms: usize,
    /// Existing generated/copied UTF-8 bytes required by the prefix.
    pub bytes: usize,
}
impl TextCapacity {
    /// Amortized next-attempt layout; conveys no allocation permission.
    pub fn grown_for(self, required: Self) -> Self {
        Self {
            atoms: grown_capacity(self.atoms, required.atoms),
            bytes: grown_capacity(self.bytes, required.bytes),
        }
    }
    /// Preserve both already required destinations across a later producer.
    pub fn union(self, other: Self) -> Self {
        Self {
            atoms: self.atoms.max(other.atoms),
            bytes: self.bytes.max(other.bytes),
        }
    }
}

/// Exact immutable generated-value slots in one fixed attempt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueCapacity {
    /// Values retained by list/map construction and materialized concatenation.
    pub slots: usize,
}
impl ValueCapacity {
    /// Preserve all destinations actually reached by completed attempts.
    pub fn union(self, other: Self) -> Self {
        Self {
            slots: self.slots.max(other.slots),
        }
    }
    /// Amortized next-attempt layout; conveys no allocation permission.
    pub fn grown_for(self, required: Self) -> Self {
        Self {
            slots: grown_capacity(self.slots, required.slots),
        }
    }
}
/// Fixed planning rejection before any render destination is allocated.
#[derive(Debug)]
pub enum RenderPlanError {
    /// The source-derived measurement destination could not be reserved.
    Reserve(TryReserveError),
    /// The next actual JSON container needs this larger fixed destination.
    JsonCapacity(JsonCapacity),
    /// The reached generated-text consumer requires this actual prefix.
    TextCapacity(TextCapacity),
    /// A reached constructor requires these additional immutable value slots.
    ValueCapacity(ValueCapacity),
    /// Checked counts or requested layouts exceed the addressable range.
    Overflow,
    /// The closed source/input state does not satisfy its checked geometry.
    Geometry,
    /// Shared-dispatch failure at its actual source instruction/location.
    Execution(RenderFailure),
}
impl RenderPlanError {
    /// True only for the actual absent-callable failure from the shared worker.
    /// Geometry, capacity, allocation, and arithmetic failures remain distinct.
    pub fn is_unknown_function(&self) -> bool {
        matches!(self, Self::Execution(failure) if failure.unknown_function)
    }
}

/// Fixed source location of a checked execution failure, copied from the
/// actually emitted image without an allocated debug object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderFailure {
    /// Program counter at which the shared instruction failed.
    pub instruction: u32,
    /// Actual emitted line when available.
    pub line: Option<usize>,
    /// Actual emitted start column when available.
    pub column: Option<usize>,
    /// Whether checked count arithmetic overflowed.
    pub overflow: bool,
    /// The actual emitted call resolved to no source/local/context/global
    /// function. Known unsupported functions are never classified this way.
    pub unknown_function: bool,
}
fn failure(source: &PreparedTemplate, error: worker::Failure) -> RenderPlanError {
    if let worker::Error::JsonCapacity(capacity) = error.kind {
        return RenderPlanError::JsonCapacity(capacity);
    }
    if let worker::Error::TextCapacity(capacity) = error.kind {
        return RenderPlanError::TextCapacity(capacity);
    }
    if let worker::Error::ValueCapacity(capacity) = error.kind {
        return RenderPlanError::ValueCapacity(capacity);
    }
    let location = source.locations.get(error.instruction as usize);
    RenderPlanError::Execution(RenderFailure {
        instruction: error.instruction,
        line: location.map(|location| location.line),
        column: location.and_then(|location| location.span.map(|span| span.start_col)),
        overflow: error.kind == worker::Error::Overflow,
        unknown_function: error.kind == worker::Error::UnknownFunction,
    })
}
impl fmt::Display for RenderPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ValueCapacity(_) => "chat value constructor needs more counted storage",
            Self::TextCapacity(_) => "chat text consumer needs more counted prefix storage",
            Self::JsonCapacity(_) => "chat JSON traversal needs more counted storage",
            Self::Reserve(_) => "chat measurement workspace reserve failed",
            Self::Overflow => "chat render layout overflow",
            Self::Geometry => "invalid closed chat render geometry",
            Self::Execution(_) => "closed chat instruction failed",
        })
    }
}
impl std::error::Error for RenderPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(error) => Some(error),
            _ => None,
        }
    }
}

/// Source-derived render destinations and the transient borrowed-input slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderBuffer {
    /// Shared-dispatch operand stack.
    Operands,
    /// Immutable generated values, retained through this rendering only.
    Values,
    /// Maximum simultaneously active source-structured loop frames.
    Frames,
    /// Maximum simultaneously active lexical local bindings.
    Locals,
    /// Every retained concatenation prefix, reused between the two executions.
    Concat,
    /// Complete rendering without an assistant generation prompt.
    WithoutPrompt,
    /// Complete rendering with an assistant generation prompt.
    WithPrompt,
    /// Nested borrowed strings copied only when a concatenation retains them.
    ContextText,
    /// Borrowed input registers; retired before a rendering or error escapes.
    Borrowed,
    /// Maximum simultaneously live namespace identities from actual source roots.
    Namespaces,
    /// Keyword/attribute destinations, with persistent borrowed-input registers.
    NamespaceFields,
    /// Ordinary recursion-limited macro invocation frames.
    MacroCalls,
    /// Separate positional/keyword binding destination before caller operands retire.
    MacroArguments,
    /// Actual shared root closure values and retained borrowed registers.
    MacroCaptures,
}
fn index(buffer: RenderBuffer) -> usize {
    match buffer {
        RenderBuffer::Operands => 0,
        RenderBuffer::Values => 13,
        RenderBuffer::Frames => 1,
        RenderBuffer::Locals => 2,
        RenderBuffer::Concat => 3,
        RenderBuffer::WithoutPrompt => 4,
        RenderBuffer::WithPrompt => 5,
        RenderBuffer::ContextText => 6,
        RenderBuffer::Borrowed => 7,
        RenderBuffer::Namespaces => 8,
        RenderBuffer::NamespaceFields => 9,
        RenderBuffer::MacroCalls => 10,
        RenderBuffer::MacroArguments => 11,
        RenderBuffer::MacroCaptures => 12,
    }
}
fn layout<T>(count: usize) -> Result<usize, RenderPlanError> {
    Layout::array::<T>(count)
        .map(|l| l.size())
        .map_err(|_| RenderPlanError::Overflow)
}

fn destination<T: Clone>(count: usize, value: T) -> Result<Vec<T>, RenderPlanError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(RenderPlanError::Reserve)?;
    values.resize(count, value);
    Ok(values)
}

/// Source/input-derived requested layouts, with no caller-supplied byte grant.
#[derive(Clone, Copy, Debug)]
pub struct RenderRequirements {
    capacities: [usize; 14],
    buffers: usize,
    required: usize,
}
impl RenderRequirements {
    /// Number of elements reserved for the actual destination.
    pub fn capacity(self, buffer: RenderBuffer) -> usize {
        self.capacities[index(buffer)]
    }
    /// Sum of requested allocation layouts, including transient input registers.
    pub fn buffer_bytes(self) -> usize {
        self.buffers
    }
    /// Requested layouts and concrete compiler/render/error controls, excluding
    /// the consumer's own Arc/account/error wrappers, which it must price.
    pub fn required_bytes(self) -> usize {
        self.required
    }
}

/// Borrows the genuine closed source and immutable messages through both runs.
/// Measuring follows the shared VM dispatcher with source-derived scratch.
#[derive(Debug)]
pub struct RenderPlan<'a> {
    source: &'a PreparedTemplate,
    context: RenderContext<'a>,
    counts: [Counts; 2],
    requirements: RenderRequirements,
    json: JsonCapacity,
    #[cfg(feature = "development-closed-chat")]
    fail: Option<RenderBuffer>,
}
impl<'a> RenderPlan<'a> {
    /// Pure source-derived measurement scratch and concrete call/error controls.
    /// This query does not walk input, allocate, or grant host capacity.
    pub fn prefix_bytes(source: &PreparedTemplate) -> Result<usize, RenderPlanError> {
        Self::prefix_bytes_with_json_capacity(source, JsonCapacity::default())
    }
    /// Pure source and explicit JSON scratch layouts before one measured attempt.
    pub fn prefix_bytes_with_json_capacity(
        source: &PreparedTemplate,
        json: JsonCapacity,
    ) -> Result<usize, RenderPlanError> {
        Self::prefix_bytes_with_capacities(source, json, TextCapacity::default())
    }
    /// Pure layout for one explicit JSON and generated-text prefix attempt.
    pub fn prefix_bytes_with_capacities(
        source: &PreparedTemplate,
        json: JsonCapacity,
        text: TextCapacity,
    ) -> Result<usize, RenderPlanError> {
        Self::prefix_bytes_with_values(source, json, text, ValueCapacity::default())
    }
    /// Pure layouts for one fixed generated-value and text attempt.
    pub fn prefix_bytes_with_values(
        source: &PreparedTemplate,
        json: JsonCapacity,
        text: TextCapacity,
        values: ValueCapacity,
    ) -> Result<usize, RenderPlanError> {
        let geometry = source.geometry;
        let namespaces = NamespaceLayout::inspect(source).ok_or(RenderPlanError::Overflow)?;
        let macros = MacroLayout::inspect(source).ok_or(RenderPlanError::Overflow)?;
        let count = macros
            .borrowed_count()
            .and_then(|n| n.checked_add(values.slots))
            .ok_or(RenderPlanError::Overflow)?;
        let parts = [
            worker::json::buffer_bytes(json).ok_or(RenderPlanError::Overflow)?,
            size_of::<JsonCapacity>(),
            size_of::<TextCapacity>(),
            size_of::<ValueCapacity>() * 2,
            size_of::<Vec<Slot>>(),
            layout::<Slot>(values.slots)?,
            size_of::<(Vec<Atom>, Vec<u8>)>(),
            size_of::<Result<Vec<Atom>, RenderPlanError>>(),
            size_of::<Result<Vec<u8>, RenderPlanError>>(),
            layout::<Atom>(text.atoms)?,
            layout::<u8>(text.bytes)?,
            size_of::<worker::json::Scratch<'_>>(),
            layout::<Slot>(geometry.operands)?,
            layout::<Frame>(geometry.frames)?,
            layout::<Local>(geometry.locals)?,
            layout::<Namespace>(namespaces.objects)?,
            layout::<NamespaceField>(namespaces.fields)?,
            layout::<CallFrame>(macros.calls)?,
            layout::<Slot>(macros.arguments)?,
            layout::<Local>(macros.closure_fields)?,
            size_of::<MacroLayout>(),
            size_of::<(Vec<CallFrame>, Vec<Slot>, Vec<Local>)>(),
            size_of::<Result<Vec<CallFrame>, RenderPlanError>>(),
            size_of::<NamespaceLayout>(),
            size_of::<Result<Vec<Namespace>, RenderPlanError>>(),
            size_of::<Result<Vec<NamespaceField>, RenderPlanError>>(),
            Borrowed::layout(count).ok_or(RenderPlanError::Overflow)?,
            size_of::<(
                Vec<Slot>,
                Vec<Frame>,
                Vec<Local>,
                Vec<Namespace>,
                Vec<NamespaceField>,
                Borrowed<'a>,
            )>(),
            size_of::<Self>(),
            size_of::<Result<Self, RenderPlanError>>(),
            size_of::<RenderRequirements>(),
            size_of::<RenderPlanError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Vec<Slot>, RenderPlanError>>(),
            size_of::<Result<Vec<Frame>, RenderPlanError>>(),
            size_of::<Result<Vec<Local>, RenderPlanError>>(),
            size_of::<[Counts; 2]>(),
            size_of::<[usize; 4]>(),
            size_of::<(JsonCapacity, JsonCapacity)>(),
            size_of::<(TextCapacity, TextCapacity)>(),
            size_of::<(ValueCapacity, ValueCapacity)>(),
            size_of::<[usize; 14]>(),
            size_of::<[usize; 14]>(),
            size_of::<Layout>(),
            size_of::<super::source::Geometry>(),
            worker::control_bytes().ok_or(RenderPlanError::Overflow)?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(RenderPlanError::Overflow)
    }

    /// Measure both actual generation-prompt settings without building ordinary
    /// Values or a renderer Environment. Reserve the pure prefix facts first
    /// when using an original host account.
    pub fn prepare(
        source: &'a PreparedTemplate,
        messages: Messages<'a>,
    ) -> Result<Self, RenderPlanError> {
        Self::prepare_context(source, RenderContext::from_messages(messages))
    }
    /// Measure with the same immutable selected external bindings used at execution.
    pub fn prepare_context(
        source: &'a PreparedTemplate,
        context: RenderContext<'a>,
    ) -> Result<Self, RenderPlanError> {
        let mut json = JsonCapacity::default();
        let mut text = TextCapacity::default();
        let mut values = ValueCapacity::default();
        loop {
            match Self::prepare_context_with_values(source, context, json, text, values) {
                Err(RenderPlanError::ValueCapacity(next)) if next.slots > values.slots => {
                    values = values.grown_for(next)
                }
                Err(RenderPlanError::TextCapacity(next)) if text.union(next) != text => {
                    text = text.grown_for(next)
                }
                Err(RenderPlanError::JsonCapacity(next)) if json.union(next) != json => {
                    json = json.grown_for(next)
                }
                result => return result,
            }
        }
    }
    /// Execute one attempt in already admitted fixed scratch. A Capacity error
    /// names the next reached destination; it never grows this attempt's Vecs.
    pub fn prepare_context_with_json_capacity(
        source: &'a PreparedTemplate,
        context: RenderContext<'a>,
        json: JsonCapacity,
    ) -> Result<Self, RenderPlanError> {
        Self::prepare_context_with_capacities(source, context, json, TextCapacity::default())
    }
    /// One fixed attempt; neither prefix destination grows during dispatch.
    pub fn prepare_context_with_capacities(
        source: &'a PreparedTemplate,
        context: RenderContext<'a>,
        json: JsonCapacity,
        text: TextCapacity,
    ) -> Result<Self, RenderPlanError> {
        Self::prepare_context_with_values(source, context, json, text, ValueCapacity::default())
    }
    /// Execute one already funded attempt without growing any value destination.
    pub fn prepare_context_with_values(
        source: &'a PreparedTemplate,
        context: RenderContext<'a>,
        json: JsonCapacity,
        text: TextCapacity,
        values: ValueCapacity,
    ) -> Result<Self, RenderPlanError> {
        Self::prefix_bytes_with_values(source, json, text, values)?;
        let mut generated_values = destination(values.slots, Slot::Undefined)?;
        let mut concat = destination(text.atoms, Atom::Source(Range::new(0, 0)))?;
        let mut context_text = destination(text.bytes, 0u8)?;
        let mut json_scratch = worker::json::prepare(json)?;
        let geometry = source.geometry;
        let mut operands = destination(geometry.operands, Slot::Undefined)?;
        let mut frames = destination(geometry.frames, Frame::default())?;
        let mut locals = destination(geometry.locals, Local::default())?;
        let namespace_layout = NamespaceLayout::inspect(source).ok_or(RenderPlanError::Overflow)?;
        let mut namespaces = destination(namespace_layout.objects, Namespace::default())?;
        let mut namespace_fields = destination(namespace_layout.fields, NamespaceField::default())?;
        let macro_layout = MacroLayout::inspect(source).ok_or(RenderPlanError::Overflow)?;
        let mut calls = destination(macro_layout.calls, CallFrame::default())?;
        let mut arguments = destination(macro_layout.arguments, Slot::Undefined)?;
        let mut closure_fields = destination(macro_layout.closure_fields, Local::default())?;
        let value_register_start = macro_layout
            .borrowed_count()
            .ok_or(RenderPlanError::Overflow)?;
        let borrowed_count = value_register_start
            .checked_add(values.slots)
            .ok_or(RenderPlanError::Overflow)?;
        let mut borrowed = Borrowed::prepare(borrowed_count, geometry.operands)
            .map_err(RenderPlanError::Reserve)?;
        let mut counts = [Counts::default(); 2];
        for (generation, count) in [false, true].into_iter().zip(&mut counts) {
            *count = worker::run(
                source,
                context,
                generation,
                Workspace {
                    operands: &mut operands,
                    values: &mut generated_values,
                    value_register_start,
                    frames: &mut frames,
                    locals: &mut locals,
                    namespaces: &mut namespaces,
                    namespace_fields: &mut namespace_fields,
                    calls: &mut calls,
                    arguments: &mut arguments,
                    closure_fields: &mut closure_fields,
                    concat: (text != TextCapacity::default()).then_some(concat.as_mut_slice()),
                    output: None,
                    context_text: (text != TextCapacity::default())
                        .then_some(context_text.as_mut_slice()),
                },
                &mut borrowed,
                &mut json_scratch,
            )
            .map_err(|error| failure(source, error))?;
        }
        let capacities = [
            geometry.operands,
            geometry.frames,
            geometry.locals,
            counts[0].concat.max(counts[1].concat),
            counts[0].output,
            counts[1].output,
            counts[0].context_text.max(counts[1].context_text),
            value_register_start
                .checked_add(counts[0].values.max(counts[1].values))
                .ok_or(RenderPlanError::Overflow)?,
            namespace_layout.objects,
            namespace_layout.fields,
            macro_layout.calls,
            macro_layout.arguments,
            macro_layout.closure_fields,
            counts[0].values.max(counts[1].values),
        ];
        let sizes = [
            layout::<Slot>(capacities[0])?,
            layout::<Frame>(capacities[1])?,
            layout::<Local>(capacities[2])?,
            layout::<Atom>(capacities[3])?,
            layout::<u8>(capacities[4])?,
            layout::<u8>(capacities[5])?,
            layout::<u8>(capacities[6])?,
            Borrowed::layout(capacities[7]).ok_or(RenderPlanError::Overflow)?,
            layout::<Namespace>(capacities[8])?,
            layout::<NamespaceField>(capacities[9])?,
            layout::<CallFrame>(capacities[10])?,
            layout::<Slot>(capacities[11])?,
            layout::<Local>(capacities[12])?,
            layout::<Slot>(capacities[13])?,
        ];
        let mut buffers = worker::json::buffer_bytes(json).ok_or(RenderPlanError::Overflow)?;
        for bytes in sizes {
            buffers = buffers
                .checked_add(bytes)
                .ok_or(RenderPlanError::Overflow)?;
        }
        let controls = [
            size_of::<Self>(),
            size_of::<Rendered>(),
            size_of::<RenderError>(),
            size_of::<RenderCause>(),
            size_of::<Result<Rendered, RenderError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<[Counts; 2]>(),
            size_of::<[usize; 4]>(),
            size_of::<(JsonCapacity, JsonCapacity)>(),
            size_of::<(TextCapacity, TextCapacity)>(),
            size_of::<(ValueCapacity, ValueCapacity)>(),
            size_of::<[usize; 14]>(), // actual capacity array
            size_of::<[usize; 14]>(), // checked byte-layout array
            size_of::<RenderRequirements>(),
            size_of::<Range>(), // resulting suffix range
            size_of::<Layout>(),
            worker::control_bytes().ok_or(RenderPlanError::Overflow)?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(RenderPlanError::Overflow)?;
        let required = buffers
            .checked_add(controls)
            .ok_or(RenderPlanError::Overflow)?
            .max(Self::prefix_bytes_with_json_capacity(source, json)?);
        Ok(Self {
            source,
            context,
            json,
            counts,
            requirements: RenderRequirements {
                capacities,
                buffers,
                required,
            },
            #[cfg(feature = "development-closed-chat")]
            fail: None,
        })
    }
    /// Actual requested layout facts before construction.
    pub fn requirements(&self) -> RenderRequirements {
        self.requirements
    }
    /// Inject only a real overflow at one selected destination reserve.
    #[cfg(feature = "development-closed-chat")]
    pub fn fail_reservation(mut self, buffer: RenderBuffer) -> Self {
        self.fail = Some(buffer);
        self
    }
    fn requested(&self, buffer: RenderBuffer) -> usize {
        #[cfg(feature = "development-closed-chat")]
        if self.fail == Some(buffer) {
            return usize::MAX;
        }
        self.requirements.capacity(buffer)
    }
    /// Reserve every destination once, then run the same finite dispatch twice.
    /// All actual prefix allocations remain in the owning error on failure.
    pub fn render(self) -> Result<Rendered, RenderError> {
        let mut data = Rendered {
            operands: Vec::new(),
            values: Vec::new(),
            frames: Vec::new(),
            locals: Vec::new(),
            namespaces: Vec::new(),
            namespace_fields: Vec::new(),
            calls: Vec::new(),
            arguments: Vec::new(),
            closure_fields: Vec::new(),
            concat: Vec::new(),
            without: Vec::new(),
            with: Vec::new(),
            context_text: Vec::new(),
            suffix: Range::new(0, 0),
        };
        macro_rules! reserve {
            ($field:ident, $buffer:ident, $empty:expr) => {
                if let Err(error) = data
                    .$field
                    .try_reserve_exact(self.requested(RenderBuffer::$buffer))
                {
                    return Err(RenderError {
                        data,
                        cause: RenderCause::Reserve(RenderBuffer::$buffer, error),
                    });
                }
                data.$field
                    .resize(self.requirements.capacity(RenderBuffer::$buffer), $empty);
            };
        }
        reserve!(operands, Operands, Slot::Undefined);
        reserve!(values, Values, Slot::Undefined);
        reserve!(frames, Frames, Frame::default());
        reserve!(locals, Locals, Local::default());
        reserve!(namespaces, Namespaces, Namespace::default());
        reserve!(namespace_fields, NamespaceFields, NamespaceField::default());
        reserve!(calls, MacroCalls, CallFrame::default());
        reserve!(arguments, MacroArguments, Slot::Undefined);
        reserve!(closure_fields, MacroCaptures, Local::default());
        reserve!(concat, Concat, Atom::Source(Range::new(0, 0)));
        reserve!(without, WithoutPrompt, 0);
        reserve!(with, WithPrompt, 0);
        reserve!(context_text, ContextText, 0);
        let mut borrowed = match Borrowed::prepare(
            self.requested(RenderBuffer::Borrowed),
            self.source.geometry.operands,
        ) {
            Ok(borrowed) => borrowed,
            Err(error) => {
                return Err(RenderError {
                    data,
                    cause: RenderCause::Reserve(RenderBuffer::Borrowed, error),
                });
            }
        };
        let mut json_scratch = match worker::json::prepare(self.json) {
            Ok(value) => value,
            Err(error) => {
                return Err(RenderError {
                    data,
                    cause: RenderCause::State(error),
                });
            }
        };
        for (index, generation) in [false, true].into_iter().enumerate() {
            // The previous run has returned with empty stack and no loop frame.
            // No slot alias survives to the next concat overwrite.
            let output = if generation {
                &mut data.with
            } else {
                &mut data.without
            };
            let result = worker::run(
                self.source,
                self.context,
                generation,
                Workspace {
                    operands: &mut data.operands,
                    values: &mut data.values,
                    value_register_start: self.requirements.capacity(RenderBuffer::Borrowed)
                        - self.requirements.capacity(RenderBuffer::Values),
                    frames: &mut data.frames,
                    locals: &mut data.locals,
                    namespaces: &mut data.namespaces,
                    namespace_fields: &mut data.namespace_fields,
                    calls: &mut data.calls,
                    arguments: &mut data.arguments,
                    closure_fields: &mut data.closure_fields,
                    concat: Some(&mut data.concat),
                    output: Some(output),
                    context_text: Some(&mut data.context_text),
                },
                &mut borrowed,
                &mut json_scratch,
            );
            match result {
                Ok(counts) if counts == self.counts[index] => {}
                Ok(_) => {
                    return Err(RenderError {
                        data,
                        cause: RenderCause::State(RenderPlanError::Geometry),
                    });
                }
                Err(error) => {
                    return Err(RenderError {
                        data,
                        cause: RenderCause::State(failure(self.source, error)),
                    });
                }
            }
        }
        data.suffix = if data.with.starts_with(&data.without) {
            Range::new(data.without.len(), data.with.len() - data.without.len())
        } else {
            Range::new(0, 0)
        };
        Ok(data)
    }
}

/// Both completed UTF-8 renderings and all original destination capacities.
/// No mutable, raw-Vec, serde, Clone or ordinary Value conversion is exposed.
#[derive(Debug)]
pub struct Rendered {
    operands: Vec<Slot>,
    values: Vec<Slot>,
    frames: Vec<Frame>,
    locals: Vec<Local>,
    namespaces: Vec<Namespace>,
    namespace_fields: Vec<NamespaceField>,
    calls: Vec<CallFrame>,
    arguments: Vec<Slot>,
    closure_fields: Vec<Local>,
    concat: Vec<Atom>,
    without: Vec<u8>,
    with: Vec<u8>,
    context_text: Vec<u8>,
    suffix: Range,
}
impl Rendered {
    /// Exact selected rendering, borrowed from the retained output destination.
    pub fn prompt(&self, generation_prompt: bool) -> &str {
        std::str::from_utf8(if generation_prompt {
            &self.with
        } else {
            &self.without
        })
        .expect("UTF-8 span copies")
    }
    /// Extra generation suffix only when the false rendering is a true prefix.
    pub fn generation_suffix(&self) -> &str {
        self.suffix.text(&self.with).expect("UTF-8 prefix suffix")
    }
    /// Actual capacities, including still-retained scratch and partial outputs.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.operands.capacity() * size_of::<Slot>()
            + self.values.capacity() * size_of::<Slot>()
            + self.frames.capacity() * size_of::<Frame>()
            + self.locals.capacity() * size_of::<Local>()
            + self.namespaces.capacity() * size_of::<Namespace>()
            + self.namespace_fields.capacity() * size_of::<NamespaceField>()
            + self.calls.capacity() * size_of::<CallFrame>()
            + self.arguments.capacity() * size_of::<Slot>()
            + self.closure_fields.capacity() * size_of::<Local>()
            + self.concat.capacity() * size_of::<Atom>()
            + self.without.capacity()
            + self.with.capacity()
            + self.context_text.capacity()
    }
}
/// Actual reserve cause or a fixed checked shared-worker state failure.
#[derive(Debug)]
pub enum RenderCause {
    /// The selected destination's own real TryReserveError.
    Reserve(RenderBuffer, TryReserveError),
    /// Checked execution state or arithmetic failed; the source stays borrowed
    /// through return, and the consumer retains its original source owner.
    State(RenderPlanError),
}
impl RenderCause {
    /// Whether actual execution rejected an absent callable, without consuming
    /// any retained partial render or changing its retirement obligations.
    pub fn is_unknown_function(&self) -> bool {
        matches!(self, Self::State(cause) if cause.is_unknown_function())
    }
}

impl fmt::Display for RenderCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("closed chat render failed")
    }
}
impl std::error::Error for RenderCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(_, error) => Some(error),
            Self::State(error) => Some(error),
        }
    }
}
/// By-value failure retaining all completed real destination prefixes.
#[derive(Debug)]
pub struct RenderError {
    data: Rendered,
    cause: RenderCause,
}
impl RenderError {
    /// Real failure without allocating an error wrapper.
    pub fn cause(&self) -> &RenderCause {
        &self.cause
    }
    /// Actual storage still retained by this owning error.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.data.retained_buffer_bytes()
    }
}
impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
