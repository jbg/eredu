//! Finite destinations for the same ordinary slice constructor.
use super::super::{TokenizerSlice, TopoNode};
use super::{Cursor, Node, SlicerSourceError, SlicerSourceView};
use crate::toktrie::{
    SimpleVob, TokTrie, TokTrieConstructionFailure, TokTrieConstructionPlan,
    TokTrieConstructionRequirements, TokTrieSourceError, TokenMaskConstructionFailure,
    TokenMaskConstructionPlan, TokenMaskSourceError,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
enum Cause {
    Source(SlicerSourceError),
    Allocation(TryReserveError),
    TrieSource(TokTrieSourceError),
    Trie(TokTrieConstructionFailure),
    MaskSource(TokenMaskSourceError),
    Mask(TokenMaskConstructionFailure),
    // Only ordinary regex recognition can create this variant. The recorded
    // constructor never enters Regex::new or a formatting path.
    Recognition(anyhow::Error),
}
impl From<SlicerSourceError> for Cause {
    fn from(e: SlicerSourceError) -> Self {
        Self::Source(e)
    }
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(e) => fmt::Display::fmt(e, f),
            Self::Allocation(e) => fmt::Display::fmt(e, f),
            Self::TrieSource(e) => fmt::Display::fmt(e, f),
            Self::Trie(e) => fmt::Display::fmt(e, f),
            Self::MaskSource(e) => fmt::Display::fmt(e, f),
            Self::Mask(e) => fmt::Display::fmt(e, f),
            Self::Recognition(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Cause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Source(e) => e,
            Self::Allocation(e) => e,
            Self::TrieSource(e) => e,
            Self::Trie(e) => e,
            Self::MaskSource(e) => e,
            Self::Mask(e) => e,
            Self::Recognition(e) => e.as_ref(),
        })
    }
}
#[derive(Default)]
struct Partial {
    idx: usize,
    regex: String,
    mask: Option<SimpleVob>,
    without: Option<SimpleVob>,
    scratch: Option<SimpleVob>,
    trimmed: Option<SimpleVob>,
    with_trie: Option<TokTrie>,
    without_trie: Option<TokTrie>,
    child_tries: Vec<TokTrie>,
    children: Vec<TokenizerSlice>,
}
impl Partial {
    fn finish(mut self) -> TokenizerSlice {
        TokenizerSlice {
            idx: self.idx,
            regex: self.regex,
            trie_without_child: self.child_tries,
            trie_without_children: self.without_trie.take().expect("completed slice"),
            trie_with_children: self.with_trie.take().expect("completed slice"),
            mask_with_children: self.mask.take().expect("completed slice"),
            mask_trimmed: self.trimmed.take().expect("completed slice"),
            children: self.children,
        }
    }
}
/// Every actual completed child and failed leaf destination remains owned until
/// the enclosing failure is destroyed; this does not release a runtime account.
pub struct SlicerConstructionFailure {
    cause: Cause,
    partials: Vec<Partial>,
    root: Option<TokenizerSlice>,
    patterns: Vec<String>,
}
impl fmt::Debug for SlicerConstructionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicerConstructionFailure")
            .field("cause", &self.cause)
            .field("partials", &self.partials.len())
            .field("has_root", &self.root.is_some())
            .field("patterns", &self.patterns.len())
            .finish()
    }
}
impl fmt::Display for SlicerConstructionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for SlicerConstructionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
#[derive(Clone, Copy)]
enum Input<'a> {
    Ordinary {
        node: &'a TopoNode,
        regexes: &'a [String],
    },
    Recorded {
        node: Node<'a>,
        source: SlicerSourceView<'a>,
    },
}
impl<'a> Input<'a> {
    fn idx(self) -> usize {
        match self {
            Self::Ordinary { node, .. } => node.value,
            Self::Recorded { node, .. } => node.idx,
        }
    }
    fn count(self) -> usize {
        match self {
            Self::Ordinary { node, .. } => node.children.len(),
            Self::Recorded { node, .. } => node.child_count,
        }
    }
    fn regex(self) -> Result<&'a str, SlicerSourceError> {
        match self {
            Self::Ordinary { node, regexes } => regexes
                .get(node.value)
                .map(String::as_str)
                .ok_or(SlicerSourceError::Record),
            Self::Recorded { node, source } => source.regex(node.idx),
        }
    }
    fn child(self, idx: usize) -> Result<Self, SlicerSourceError> {
        match self {
            Self::Ordinary { node, regexes } => Ok(Self::Ordinary {
                node: node.children.get(idx).ok_or(SlicerSourceError::Record)?,
                regexes,
            }),
            Self::Recorded { node, source } => {
                if idx >= node.child_count {
                    return Err(SlicerSourceError::Record);
                }
                let mut cursor = Cursor::new(node.children);
                for current in 0..=idx {
                    let mut head = cursor;
                    let bytes = cursor.take(head.usize()?)?;
                    if current == idx {
                        return Ok(Self::Recorded {
                            node: Node::parse(bytes)?,
                            source,
                        });
                    }
                }
                Err(SlicerSourceError::Record)
            }
        }
    }
}
#[derive(Clone, Copy)]
struct Allowance {
    masks: usize,
    filters: usize,
    mask_bytes: usize,
    filter_bytes: usize,
    string_bytes: usize,
    child_slots: usize,
}
struct Context {
    allowance: Option<Allowance>,
    partials: Vec<Partial>,
}
impl Context {
    fn mask(&mut self, plan: TokenMaskConstructionPlan<'_>) -> Result<SimpleVob, Cause> {
        if let Some(a) = &mut self.allowance {
            if plan.requirements().required_bytes() > a.mask_bytes || a.masks == 0 {
                return Err(SlicerSourceError::Destination.into());
            }
            a.masks -= 1;
        }
        plan.compile().map_err(Cause::Mask)
    }
    fn empty_mask(&mut self, trie: &TokTrie) -> Result<SimpleVob, Cause> {
        self.mask(TokenMaskConstructionPlan::for_trie(trie).map_err(Cause::MaskSource)?)
    }
    fn copy_mask(&mut self, mask: &SimpleVob) -> Result<SimpleVob, Cause> {
        self.mask(TokenMaskConstructionPlan::copy(mask).map_err(Cause::MaskSource)?)
    }
    fn filter(&mut self, trie: &TokTrie, mask: &SimpleVob) -> Result<TokTrie, Cause> {
        let plan =
            TokTrieConstructionPlan::prepare_filter(trie, mask).map_err(Cause::TrieSource)?;
        if let Some(a) = &mut self.allowance {
            if plan.requirements().required_bytes() > a.filter_bytes || a.filters == 0 {
                return Err(SlicerSourceError::Destination.into());
            }
            a.filters -= 1;
        }
        plan.compile().map_err(Cause::Trie)
    }
    fn string(&mut self, output: &mut String, value: &str) -> Result<(), Cause> {
        if let Some(a) = &mut self.allowance {
            a.string_bytes = a
                .string_bytes
                .checked_sub(value.len())
                .ok_or(SlicerSourceError::Destination)?;
        }
        output
            .try_reserve_exact(value.len())
            .map_err(Cause::Allocation)?;
        if output.capacity() > value.len() {
            return Err(SlicerSourceError::Destination.into());
        }
        output.push_str(value);
        Ok(())
    }
    fn child_slots(&mut self, partial: &mut Partial, n: usize) -> Result<(), Cause> {
        if let Some(a) = &mut self.allowance {
            a.child_slots = a
                .child_slots
                .checked_sub(n)
                .ok_or(SlicerSourceError::Destination)?;
        }
        reserve(&mut partial.children, n)?;
        reserve(&mut partial.child_tries, n)
    }
    fn keep(&mut self, partial: Partial) {
        // Paid callers reserve one failure frame per actual tree depth before
        // any child construction. Ordinary failures may grow their host vector.
        assert!(self.allowance.is_none() || self.partials.len() < self.partials.capacity());
        self.partials.push(partial);
    }
}
fn reserve<T>(output: &mut Vec<T>, n: usize) -> Result<(), Cause> {
    output.try_reserve_exact(n).map_err(Cause::Allocation)?;
    if output.capacity() > n {
        return Err(SlicerSourceError::Destination.into());
    }
    Ok(())
}
fn build(context: &mut Context, trie: &TokTrie, input: Input<'_>) -> Result<TokenizerSlice, Cause> {
    let mut partial = Partial {
        idx: input.idx(),
        ..Partial::default()
    };
    let result = (|| {
        context.string(&mut partial.regex, input.regex()?)?;
        partial.mask = Some(context.empty_mask(trie)?);
        let mask = partial.mask.as_mut().expect("initialized mask");
        match input {
            Input::Ordinary { .. } => {
                if context.allowance.is_some() {
                    return Err(SlicerSourceError::Record.into());
                }
                if partial.regex.is_empty() {
                    mask.set_all(true);
                } else {
                    let mut regex = crate::derivre::Regex::new(&partial.regex).map_err(|e| {
                        Cause::Recognition(anyhow::anyhow!(
                            "invalid regex: {:?}: {}",
                            partial.regex,
                            e
                        ))
                    })?;
                    for id in 0..trie.vocab_size() as u32 {
                        let bytes = trie.token(id);
                        if !bytes.is_empty() && regex.is_match_bytes(bytes) {
                            mask.allow_token(id);
                        }
                    }
                }
            }
            Input::Recorded { node, .. } => {
                if node.words.len() / 4 != mask.as_slice().len() {
                    return Err(SlicerSourceError::Record.into());
                }
                for (word_idx, bytes) in node.words.chunks_exact(4).enumerate() {
                    let word = u32::from_le_bytes(
                        bytes.try_into().map_err(|_| SlicerSourceError::Record)?,
                    );
                    for bit in 0..32 {
                        if word & (1u32 << bit) != 0 {
                            mask.allow_token((word_idx * 32 + bit) as u32);
                        }
                    }
                }
            }
        }
        partial.with_trie =
            Some(context.filter(trie, partial.mask.as_ref().expect("initialized mask"))?);
        partial.without =
            Some(context.copy_mask(partial.mask.as_ref().expect("initialized mask"))?);
        context.child_slots(&mut partial, input.count())?;
        for idx in 0..input.count() {
            let child = build(context, trie, input.child(idx)?)?;
            partial.children.push(child);
            let child = partial.children.last().expect("inserted child");
            partial.scratch =
                Some(context.copy_mask(partial.mask.as_ref().expect("initialized mask"))?);
            partial
                .scratch
                .as_mut()
                .expect("initialized mask")
                .sub(&child.mask_with_children);
            partial.child_tries.push(context.filter(
                partial.with_trie.as_ref().expect("initialized trie"),
                partial.scratch.as_ref().expect("initialized mask"),
            )?);
            partial
                .without
                .as_mut()
                .expect("initialized mask")
                .sub(&child.mask_with_children);
            partial.scratch = None;
        }
        partial.without_trie =
            Some(context.filter(trie, partial.without.as_ref().expect("initialized mask"))?);
        partial.trimmed =
            Some(context.copy_mask(partial.mask.as_ref().expect("initialized mask"))?);
        partial
            .trimmed
            .as_mut()
            .expect("initialized mask")
            .trim_trailing_zeros();
        Ok(())
    })();
    match result {
        Ok(()) => Ok(partial.finish()),
        Err(cause) => {
            context.keep(partial);
            Err(cause)
        }
    }
}

pub(in crate::earley::slicer) fn ordinary_from_topo(
    node: &TopoNode,
    trie: &TokTrie,
    regexes: &[String],
) -> anyhow::Result<TokenizerSlice> {
    let mut context = Context {
        allowance: None,
        partials: Vec::new(),
    };
    build(&mut context, trie, Input::Ordinary { node, regexes }).map_err(|cause| {
        SlicerConstructionFailure {
            cause,
            partials: context.partials,
            root: None,
            patterns: Vec::new(),
        }
        .into()
    })
}
pub(in crate::earley::slicer) fn ordinary_from_record(
    node: Node<'_>,
    trie: &TokTrie,
    source: SlicerSourceView<'_>,
) -> anyhow::Result<TokenizerSlice> {
    let mut context = Context {
        allowance: None,
        partials: Vec::new(),
    };
    build(&mut context, trie, Input::Recorded { node, source }).map_err(|cause| {
        SlicerConstructionFailure {
            cause,
            partials: context.partials,
            root: None,
            patterns: Vec::new(),
        }
        .into()
    })
}

/// Complete source-derived constructor envelope. Filtered tries reserve at most
/// the actual full vocabulary's constructor geometry; no default arena is used.
#[derive(Clone, Copy)]
pub struct SlicerConstructionRequirements {
    allowance: Allowance,
    depth: usize,
    pattern_count: usize,
    buffers: usize,
    controls: usize,
    total: usize,
    mask_controls: usize,
}
impl fmt::Debug for SlicerConstructionRequirements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicerConstructionRequirements")
            .field("buffers", &self.buffers)
            .field("controls", &self.controls)
            .field("depth", &self.depth)
            .finish()
    }
}
impl SlicerConstructionRequirements {
    /// Retained and temporary buffers, including failed parent rows.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named constructor and fixed recursive call representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete accepted envelope before the first destination allocation.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// One actual record/trie loan. It cannot be constructed from caller capacities.
pub struct SlicerConstructionPlan<'a> {
    source: SlicerSourceView<'a>,
    trie: &'a TokTrie,
    requirements: SlicerConstructionRequirements,
}
impl fmt::Debug for SlicerConstructionPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicerConstructionPlan")
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}
fn rejected(cause: Cause) -> SlicerConstructionFailure {
    SlicerConstructionFailure {
        cause,
        partials: Vec::new(),
        root: None,
        patterns: Vec::new(),
    }
}
fn extent<T>(count: usize) -> Result<usize, Cause> {
    Layout::array::<T>(count)
        .map(|v| v.size())
        .map_err(|_| SlicerSourceError::Overflow.into())
}
fn mul(a: usize, b: usize) -> Result<usize, Cause> {
    a.checked_mul(b)
        .ok_or_else(|| SlicerSourceError::Overflow.into())
}
fn sum(a: usize, b: usize) -> Result<usize, Cause> {
    a.checked_add(b)
        .ok_or_else(|| SlicerSourceError::Overflow.into())
}
fn string_geometry(node: Node<'_>, source: SlicerSourceView<'_>) -> Result<usize, Cause> {
    let mut bytes = source.regex(node.idx)?.len();
    let input = Input::Recorded { node, source };
    for idx in 0..node.child_count {
        let Input::Recorded { node: child, .. } = input.child(idx)? else {
            return Err(SlicerSourceError::Record.into());
        };
        bytes = sum(bytes, string_geometry(child, source)?)?;
    }
    Ok(bytes)
}
impl<'a> SlicerConstructionPlan<'a> {
    /// Source inspection frames to fund before traversing the retained record.
    /// This reads only the closed descriptor's finite depth, with no allocation.
    pub fn inspection_control_bytes(source: SlicerSourceView<'_>) -> Option<usize> {
        let recursive = [
            size_of::<Input<'_>>(),
            size_of::<Node<'_>>(),
            size_of::<Cursor<'_>>(),
            size_of::<(Node<'_>, SlicerSourceView<'_>)>(),
            size_of::<Result<usize, Cause>>(),
            size_of::<std::ops::Range<usize>>(),
        ];
        let frames = recursive
            .into_iter()
            .try_fold(size_of_val(&recursive), usize::checked_add)?
            .checked_mul(source.layout.depth)?;
        let parts = [
            frames,
            super::SlicerSourceDescriptor(source.layout).control_bytes()?,
            size_of::<Self>(),
            size_of::<SlicerConstructionRequirements>(),
            size_of::<Cause>(),
            size_of::<SlicerConstructionFailure>(),
            size_of::<SlicerSourceView<'_>>(),
            size_of::<Result<Self, SlicerConstructionFailure>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<TokTrieConstructionRequirements>(),
            size_of::<crate::toktrie::TokenMaskConstructionRequirements>(),
            size_of::<(&SlicerSourceView<'_>, &TokTrie)>(),
            size_of::<Result<usize, Cause>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Cursor<'_>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Derives bounds from the actual vocabulary and closed recorded tree. Source
    /// registration/account identity is still the enclosing owner's contract.
    pub fn prepare(
        source: SlicerSourceView<'a>,
        trie: &'a TokTrie,
    ) -> Result<Self, SlicerConstructionFailure> {
        Self::inspect(source, trie).map_err(rejected)
    }
    fn inspect(source: SlicerSourceView<'a>, trie: &'a TokTrie) -> Result<Self, Cause> {
        if !source.matches_trie(trie) {
            return Err(SlicerSourceError::Tokenizer.into());
        }
        let root = Node::parse(&source.bytes[source.layout.root_offset..])?;
        let nodes = source.layout.nodes;
        if nodes == 0 || source.layout.depth == 0 {
            return Err(SlicerSourceError::Record.into());
        }
        let edges = nodes - 1;
        let masks = sum(mul(nodes, 3)?, edges)?;
        let filters = sum(mul(nodes, 2)?, edges)?;
        let mut token_bytes = 0;
        let mut maximum = 0;
        for id in 0..trie.vocab_size() {
            let len = trie.token(id as u32).len();
            token_bytes = sum(token_bytes, len)?;
            maximum = maximum.max(len);
        }
        let filter = TokTrieConstructionRequirements::for_source_geometry(
            trie.vocab_size(),
            token_bytes,
            maximum,
            trie.eos_tokens().len(),
        )
        .map_err(Cause::TrieSource)?;
        let mask = TokenMaskConstructionPlan::for_trie(trie)
            .map_err(Cause::MaskSource)?
            .requirements();
        let mut patterns =
            Cursor::new(&source.bytes[source.layout.source_end..source.layout.root_offset]);
        let pattern_count = patterns.usize()?;
        let mut string_bytes = string_geometry(root, source)?;
        for _ in 0..pattern_count {
            string_bytes = sum(string_bytes, patterns.string()?.len())?;
        }
        if !patterns.remaining().is_empty() {
            return Err(SlicerSourceError::Record.into());
        }
        let buffer_parts = [
            mul(masks, mask.buffer_bytes())?,
            mul(filters, filter.buffer_bytes())?,
            extent::<TokenizerSlice>(edges)?,
            extent::<TokTrie>(edges)?,
            extent::<String>(pattern_count)?,
            extent::<Partial>(source.layout.depth)?,
            string_bytes,
        ];
        let buffers = buffer_parts.into_iter().try_fold(0, sum)?;
        let recursive_parts = [
            size_of::<Partial>(),
            size_of::<Input<'_>>(),
            size_of::<Node<'_>>(),
            size_of::<Cursor<'_>>(),
            size_of::<(&mut Context, &TokTrie, Input<'_>)>(),
            size_of::<Result<TokenizerSlice, Cause>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<(Node<'_>, SlicerSourceView<'_>)>(),
            size_of::<Result<usize, Cause>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::slice::ChunksExact<'_, u8>>(),
            size_of::<(usize, u32, usize)>(),
        ];
        let recursion = mul(
            recursive_parts
                .into_iter()
                .try_fold(size_of_val(&recursive_parts), sum)?,
            source.layout.depth,
        )?;
        let parts = [
            super::SlicerSourceDescriptor(source.layout)
                .control_bytes()
                .ok_or(SlicerSourceError::Overflow)?,
            mul(masks, mask.control_bytes())?,
            mul(filters, filter.control_bytes())?,
            recursion,
            size_of_val(&buffer_parts),
            size_of::<Self>(),
            size_of::<SlicerConstructionRequirements>(),
            size_of::<SlicerConstructionFailure>(),
            size_of::<SlicerProgram>(),
            size_of::<Context>(),
            size_of::<Allowance>(),
            size_of::<Cause>(),
            size_of::<Vec<Partial>>(),
            size_of::<Vec<String>>(),
            size_of::<Option<TokenizerSlice>>(),
            size_of::<TokenizerSlice>(),
            size_of::<String>(),
            size_of::<Result<Self, SlicerConstructionFailure>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<SlicerProgram, SlicerConstructionFailure>>(),
            size_of::<Result<SlicerProgram, Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<TokTrieConstructionRequirements>(),
            size_of::<crate::toktrie::TokenMaskConstructionRequirements>(),
            size_of::<(SlicerSourceView<'_>, &TokTrie)>(),
            size_of::<Cursor<'_>>(),
            size_of::<Result<&str, SlicerSourceError>>(),
            size_of::<(&mut String, &str)>(),
            size_of::<(&mut Vec<String>, usize)>(),
            size_of::<(&mut Vec<Partial>, usize)>(),
            size_of::<(&mut Vec<TokTrie>, usize)>(),
            size_of::<(&mut Vec<TokenizerSlice>, usize)>(),
            size_of::<Result<usize, Cause>>(),
        ];
        let controls = parts.into_iter().try_fold(size_of_val(&parts), sum)?;
        Ok(Self {
            source,
            trie,
            requirements: SlicerConstructionRequirements {
                allowance: Allowance {
                    masks,
                    filters,
                    mask_bytes: mask.required_bytes(),
                    filter_bytes: filter.required_bytes(),
                    string_bytes,
                    child_slots: edges,
                },
                depth: source.layout.depth,
                pattern_count,
                mask_controls: mask.control_bytes(),
                buffers,
                controls,
                total: sum(buffers, controls)?,
            },
        })
    }
    /// Complete immutable quote for this same source loan.
    pub fn requirements(&self) -> SlicerConstructionRequirements {
        self.requirements
    }
    /// Constructs the same slice tree with one finite failed-parent table. This
    /// only closes factory storage; per-step parser/lexer/mask admission is separate.
    pub fn compile(self) -> Result<SlicerProgram, SlicerConstructionFailure> {
        let mut context = Context {
            allowance: Some(self.requirements.allowance),
            partials: Vec::new(),
        };
        let mut root = None;
        let mut patterns = Vec::new();
        let result = (|| {
            reserve(&mut context.partials, self.requirements.depth)?;
            root = Some(build(
                &mut context,
                self.trie,
                Input::Recorded {
                    node: Node::parse(&self.source.bytes[self.source.layout.root_offset..])?,
                    source: self.source,
                },
            )?);
            reserve(&mut patterns, self.requirements.pattern_count)?;
            let mut input = Cursor::new(
                &self.source.bytes[self.source.layout.source_end..self.source.layout.root_offset],
            );
            if input.usize()? != self.requirements.pattern_count {
                return Err(SlicerSourceError::Record.into());
            }
            for _ in 0..self.requirements.pattern_count {
                patterns.push(String::new());
                context.string(
                    patterns.last_mut().expect("inserted pattern"),
                    input.string()?,
                )?;
            }
            if !input.remaining().is_empty() {
                return Err(SlicerSourceError::Record.into());
            }
            let remaining = context.allowance.expect("paid constructor");
            if remaining.masks != 0
                || remaining.filters != 0
                || remaining.string_bytes != 0
                || remaining.child_slots != 0
            {
                return Err(SlicerSourceError::Destination.into());
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(SlicerProgram {
                root: root.expect("completed slicer"),
                patterns,
                step_depth: self.requirements.depth,
                step_mask_controls: self.requirements.mask_controls,
            }),
            Err(cause) => Err(SlicerConstructionFailure {
                cause,
                partials: context.partials,
                root,
                patterns,
            }),
        }
    }
}
/// Originally constructed immutable slice tree. It owns no tokenizer callback,
/// regex compiler or account; the enclosing owner retains actual source custody.
pub struct SlicerProgram {
    root: TokenizerSlice,
    patterns: Vec<String>,
    step_depth: usize,
    step_mask_controls: usize,
}
impl fmt::Debug for SlicerProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicerProgram")
            .field("patterns", &self.patterns.len())
            .finish_non_exhaustive()
    }
}
impl SlicerProgram {
    /// Fixed inspection frames before reading this actual immutable tree.
    pub fn step_inspection_control_bytes(&self) -> Option<usize> {
        super::super::step::SlicerStepPlan::inspection_control_bytes(self.step_depth)?
            .checked_add(self.step_mask_controls)
    }
    /// Exact local mask and traversal destinations; parser/lexer admission is separate.
    pub fn step_plan(
        &self,
    ) -> Result<super::super::step::SlicerStepPlan<'_>, super::super::step::SlicerStepFailure> {
        super::super::step::SlicerStepPlan::prepare(&self.root, self.trie())
    }
    /// Exact retained additional lexemes, borrowed without an ordinary clone.
    pub fn extra_lexemes(&self) -> &[String] {
        &self.patterns
    }
    /// Complete wildcard-root trie produced by the same filtered constructor.
    pub fn trie(&self) -> &TokTrie {
        &self.root.trie_with_children
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earley::SlicedBiasComputer;
    use crate::toktrie::ApproximateTokEnv;
    #[test]
    fn paid_slicer_uses_shared_tree_and_retains_actual_failed_parent_prefix() {
        let env = ApproximateTokEnv::single_byte_env();
        let ordinary = SlicedBiasComputer::new(&env, &["[a-z]+".into(), " +".into()]).unwrap();
        let source = ordinary.source_plan().unwrap().compile().unwrap();
        let plan = SlicerConstructionPlan::prepare(source.view(), env.tok_trie()).unwrap();
        assert!(plan.requirements().required_bytes() > plan.requirements().buffer_bytes());
        let program = plan.compile().unwrap();
        assert_eq!(program.extra_lexemes(), ordinary.extra_lexemes());
        fn compare(a: &TokenizerSlice, b: &TokenizerSlice) {
            assert_eq!(a.idx, b.idx);
            assert_eq!(a.regex, b.regex);
            assert_eq!(a.mask_with_children, b.mask_with_children);
            assert_eq!(a.mask_trimmed, b.mask_trimmed);
            assert_eq!(
                a.trie_with_children.sorted_tokens(),
                b.trie_with_children.sorted_tokens()
            );
            assert_eq!(
                a.trie_without_children.sorted_tokens(),
                b.trie_without_children.sorted_tokens()
            );
            assert_eq!(a.trie_without_child.len(), b.trie_without_child.len());
            for (a, b) in a.trie_without_child.iter().zip(&b.trie_without_child) {
                assert_eq!(a.sorted_tokens(), b.sorted_tokens());
            }
            assert_eq!(a.children.len(), b.children.len());
            for (a, b) in a.children.iter().zip(&b.children) {
                compare(a, b);
            }
        }
        compare(&program.root, &ordinary.top_slice);
        let letters = program
            .root
            .children
            .iter()
            .find(|child| child.idx == 0)
            .unwrap();
        let spaces = program
            .root
            .children
            .iter()
            .find(|child| child.idx == 1)
            .unwrap();
        assert_eq!(letters.mask_with_children.num_set(), 26);
        assert_eq!(spaces.mask_with_children.num_set(), 1);
        for id in 0..program.trie().vocab_size() as u32 {
            assert_eq!(
                letters.mask_with_children.is_allowed(id),
                (b'a' as u32..=b'z' as u32).contains(&id)
            );
            assert_eq!(spaces.mask_with_children.is_allowed(id), id == b' ' as u32);
        }

        let mut stopped = SlicerConstructionPlan::prepare(source.view(), env.tok_trie()).unwrap();
        stopped.requirements.allowance.filters = 4;
        let error = stopped.compile().unwrap_err();
        assert!(error.partials.len() >= 2);
        assert!(error.partials.iter().any(|p| p.with_trie.is_some()));
        assert!(error.partials.iter().any(|p| !p.children.is_empty()));
        assert!(error.partials.iter().any(|p| p.mask.is_some()));
        let token = program.trie().token(97).to_vec();
        drop((source, ordinary, env));
        assert_eq!(program.trie().token(97), token);
        assert!(error.partials.iter().any(|p| p.with_trie.is_some()));
    }
}
