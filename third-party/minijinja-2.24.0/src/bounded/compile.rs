//! Source-bound construction using the existing checked syntax and VM storage.
#![forbid(unsafe_code)]
use super::{
    source::*,
    template_syntax::{self, WhitespaceConfig, emit},
};
use std::{alloc::Layout, fmt, mem::size_of};

/// A genuine decoded-byte source iterator and selected name. Inspection only
/// traverses a clone; all decoding, syntax, continuation and output reserves
/// happen once in `compile`, after the consumer admits these requirements.
pub struct SourceCompilePlan<'a, I> {
    input: I,
    name: TemplateName<'a>,
    whitespace: WhitespaceConfig,
    source_bytes: usize,
    syntax_heap: usize,
    code: emit::Requirements,
    requirements: CompileRequirements,
    #[cfg(feature = "development-closed-chat")]
    fail: Option<SourceBuffer>,
}
impl<I> fmt::Debug for SourceCompilePlan<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceCompilePlan")
            .field("source_bytes", &self.source_bytes)
            .field("requirements", &self.requirements)
            .finish()
    }
}
impl<'a, I: Iterator<Item = u8> + Clone> SourceCompilePlan<'a, I> {
    /// Inspect the actual encoded/UTF-8 source through its allocation-free byte
    /// iterator. The consumer must supply a deterministic, allocation-free clone
    /// and traversal (the text adapter uses only UTF-8 and validated JSON byte
    /// views). Arbitrary iterator implementations receive no such guarantee.
    /// This is a source compiler plan, never an accounting grant.
    pub fn prepare(
        input: I,
        name: TemplateName<'a>,
        settings: TemplateSettings,
    ) -> Result<Self, SourceError> {
        if !settings.matches() || name.escape() != crate::AutoEscape::None {
            return Err(SourceError::Settings);
        }
        let source_bytes = input
            .clone()
            .try_fold(0usize, |n, _| n.checked_add(1))
            .ok_or(SourceError::Overflow)?;
        let syntax =
            template_syntax::byte_requirements(source_bytes).map_err(|_| SourceError::Overflow)?;
        let code = emit::requirements(syntax.records(), syntax.statements())?;
        let bytes = source_bytes
            .checked_add(name.byte_len()?)
            .ok_or(SourceError::Overflow)?;
        let arrays = [
            Layout::array::<u8>(source_bytes),
            Layout::array::<u8>(bytes),
            Layout::array::<Instruction>(code.instructions),
            Layout::array::<Location>(code.instructions),
        ];
        let buffers = arrays.into_iter().try_fold(
            syntax
                .heap_bytes()
                .checked_add(code.heap)
                .ok_or(SourceError::Overflow)?,
            |n, l| {
                n.checked_add(l.map_err(|_| SourceError::Overflow)?.size())
                    .ok_or(SourceError::Overflow)
            },
        )?;
        let controls = [
            size_of::<Self>(),
            size_of::<I>(),
            size_of::<Option<I>>(),
            size_of::<PreparedTemplate>(),
            size_of::<Result<PreparedTemplate, CompileError>>(),
            size_of::<CompileError>(),
            size_of::<CompileCause>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<CompileRequirements>(),
            size_of::<Layout>(),
            size_of::<[Layout; 4]>(),
            size_of::<[&str; 2]>(),
            size_of::<WhitespaceConfig>(),
            syntax.plan_control_bytes(),
            syntax.retained_control_bytes(),
            syntax.worker_control_bytes(),
            code.controls,
        ];
        let required = controls
            .into_iter()
            .try_fold(buffers, usize::checked_add)
            .ok_or(SourceError::Overflow)?;
        Ok(Self {
            input,
            name,
            whitespace: WhitespaceConfig {
                trim_blocks: settings.trim_blocks,
                lstrip_blocks: settings.lstrip_blocks,
                keep_trailing_newline: settings.keep_trailing_newline,
            },
            source_bytes,
            syntax_heap: syntax.heap_bytes(),
            code,
            requirements: CompileRequirements {
                capacities: [code.instructions, bytes, code.instructions],
                buffers,
                required,
            },
            #[cfg(feature = "development-closed-chat")]
            fail: None,
        })
    }
    /// Complete compiler temporary/output layouts before any reserve.
    pub fn requirements(&self) -> CompileRequirements {
        self.requirements
    }
    /// Inject an actual output reserve failure for focused custody verification.
    #[cfg(feature = "development-closed-chat")]
    pub fn fail_reservation(mut self, buffer: SourceBuffer) -> Self {
        self.fail = Some(buffer);
        self
    }
    fn requested(&self, buffer: SourceBuffer, ordinary: usize) -> usize {
        #[cfg(feature = "development-closed-chat")]
        if self.fail == Some(buffer) {
            return usize::MAX;
        }
        let _ = buffer;
        ordinary
    }
    /// Compile once. Syntax buffers are local and retire before returning;
    /// partial output/decoded buffers remain in the owning error. No references
    /// into syntax or the input iterator escape the successful output.
    pub fn compile(self) -> Result<PreparedTemplate, CompileError> {
        let mut data = PreparedTemplate {
            instructions: Vec::new(),
            bytes: Vec::new(),
            locations: Vec::new(),
            decoded_source: Some(Vec::new()),
            name: Range::new(0, 0),
            geometry: Geometry {
                operands: 3,
                frames: 1,
                locals: 1,
            },
        };
        macro_rules! reserve {
            ($field:expr,$buffer:ident,$count:expr) => {
                if let Err(error) =
                    $field.try_reserve_exact(self.requested(SourceBuffer::$buffer, $count))
                {
                    return Err(CompileError {
                        data,
                        cause: CompileCause::Reserve(SourceBuffer::$buffer, error),
                    });
                }
            };
        }
        reserve!(
            data.decoded_source.as_mut().unwrap(),
            Bytes,
            self.source_bytes
        );
        for byte in self.input.clone() {
            let source = data.decoded_source.as_mut().unwrap();
            if source.len() == self.source_bytes {
                return Err(CompileError {
                    data,
                    cause: CompileCause::Source(SourceError::Geometry),
                });
            }
            source.push(byte);
        }
        if data.decoded_source.as_ref().unwrap().len() != self.source_bytes {
            return Err(CompileError {
                data,
                cause: CompileCause::Source(SourceError::Geometry),
            });
        }
        reserve!(data.instructions, Instructions, self.code.instructions);
        reserve!(
            data.bytes,
            Bytes,
            self.requirements.capacity(SourceBuffer::Bytes)
        );
        reserve!(data.locations, Locations, self.code.instructions);
        let result = (|| -> Result<(), CompileCause> {
            let text = std::str::from_utf8(data.decoded_source.as_ref().unwrap())
                .map_err(|_| CompileCause::Source(SourceError::Syntax))?;
            // Filename is only borrowed diagnostics during syntax construction;
            // the actual composed selected name is copied into final storage.
            let plan = template_syntax::Plan::inspect(text, self.name.parts()[0], self.whitespace)
                .map_err(|_| CompileCause::Source(SourceError::Overflow))?;
            if plan.requirements().heap_bytes() > self.syntax_heap {
                return Err(CompileCause::Source(SourceError::Geometry));
            }
            let syntax = plan
                .construct()
                .map_err(|error| error.into_compile_cause())?;
            data.geometry = emit::compile(
                &syntax,
                &mut data.instructions,
                &mut data.bytes,
                &mut data.locations,
                self.code,
                self.source_bytes,
            )?;
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(CompileError { data, cause });
        }
        let start = data.bytes.len();
        for part in self.name.parts() {
            let Some(end) = data.bytes.len().checked_add(part.len()) else {
                return Err(CompileError {
                    data,
                    cause: CompileCause::Source(SourceError::Overflow),
                });
            };
            if end > self.requirements.capacity(SourceBuffer::Bytes) || end > data.bytes.capacity()
            {
                return Err(CompileError {
                    data,
                    cause: CompileCause::Source(SourceError::Geometry),
                });
            }
            data.bytes.extend_from_slice(part.as_bytes());
        }
        data.name = Range::new(start, data.bytes.len() - start);
        Ok(data)
    }
}
