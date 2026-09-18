//! Slicer-local per-step destinations. Parser/lexer storage is a separate contract.
use super::TokenizerSlice;
use crate::{
    earley::ParserRecognizer,
    toktrie::{
        SimpleVob, TokTrie, TokenMaskConstructionFailure, TokenMaskConstructionPlan,
        TokenMaskConstructionRequirements, TokenMaskSourceError,
    },
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug, Clone, Copy)]
struct Geometry {
    flags: usize,
    depth: usize,
}
fn geometry(root: &TokenizerSlice) -> Option<Geometry> {
    let mut suffix = 0;
    let mut depth = 0;
    for child in &root.children {
        let g = geometry(child)?;
        suffix = suffix.max(g.flags);
        depth = depth.max(g.depth);
    }
    Some(Geometry {
        flags: root.children.len().checked_add(suffix)?,
        depth: depth.checked_add(1)?,
    })
}
/// Exact local destinations from the same immutable slice tree and vocabulary.
#[derive(Debug, Clone, Copy)]
pub struct SlicerStepRequirements {
    flags: usize,
    buffers: usize,
    controls: usize,
    total: usize,
}
impl SlicerStepRequirements {
    /// Maximum simultaneously live child-match flags along an actual tree path.
    pub fn match_flags(&self) -> usize {
        self.flags
    }
    /// Mask words plus child-match storage; excludes parser/lexer destinations.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed local source, constructor, recursion and failure frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete slicer-local quote, not a parser or controller admission.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Closed source loan. A caller cannot substitute capacities or a different tree.
pub struct SlicerStepPlan<'a> {
    root: &'a TokenizerSlice,
    mask: TokenMaskConstructionPlan<'a>,
    requirements: SlicerStepRequirements,
}
impl fmt::Debug for SlicerStepPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicerStepPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
#[derive(Debug)]
enum Cause {
    Overflow,
    Capacity,
    Allocation(TryReserveError),
    MaskSource(TokenMaskSourceError),
    Mask(TokenMaskConstructionFailure),
}
/// Retains the actual mask and flag allocation prefix on construction failure.
#[derive(Debug)]
pub struct SlicerStepFailure {
    cause: Cause,
    mask: Option<SimpleVob>,
    flags: Vec<bool>,
}
impl fmt::Display for SlicerStepFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("slicer step geometry overflow"),
            Cause::Capacity => f.write_str("slicer step allocation exceeded its exact extent"),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
            Cause::MaskSource(e) => fmt::Display::fmt(e, f),
            Cause::Mask(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for SlicerStepFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Allocation(e) => Some(e),
            Cause::MaskSource(e) => Some(e),
            Cause::Mask(e) => Some(e),
            _ => None,
        }
    }
}
fn rejected(cause: Cause) -> SlicerStepFailure {
    SlicerStepFailure {
        cause,
        mask: None,
        flags: Vec::new(),
    }
}
impl<'a> SlicerStepPlan<'a> {
    pub(super) fn prepare(
        root: &'a TokenizerSlice,
        trie: &'a TokTrie,
    ) -> Result<Self, SlicerStepFailure> {
        Self::inspect(root, trie).map_err(rejected)
    }
    fn inspect(root: &'a TokenizerSlice, trie: &'a TokTrie) -> Result<Self, Cause> {
        let g = geometry(root).ok_or(Cause::Overflow)?;
        let mask = TokenMaskConstructionPlan::for_trie(trie).map_err(Cause::MaskSource)?;
        let flags = Layout::array::<bool>(g.flags)
            .map_err(|_| Cause::Overflow)?
            .size();
        let buffers = flags
            .checked_add(mask.requirements().buffer_bytes())
            .ok_or(Cause::Overflow)?;
        let controls = Self::inspection_control_bytes(g.depth)
            .and_then(|n| n.checked_add(mask.requirements().control_bytes()))
            .ok_or(Cause::Overflow)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        Ok(Self {
            root,
            mask,
            requirements: SlicerStepRequirements {
                flags: g.flags,
                buffers,
                controls,
                total,
            },
        })
    }
    pub(super) fn inspection_control_bytes(depth: usize) -> Option<usize> {
        let recursive = [
            size_of::<Geometry>(),
            size_of::<(
                &TokenizerSlice,
                &mut ParserRecognizer<'_>,
                &mut SimpleVob,
                &mut [bool],
            )>(),
            size_of::<(&mut [bool], &mut [bool])>(),
            size_of::<std::slice::Iter<'_, TokenizerSlice>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, TokenizerSlice>>>(),
            size_of::<Option<Geometry>>(),
            size_of::<Option<usize>>(),
            size_of::<(usize, usize, bool)>(),
        ];
        let recursion = recursive
            .into_iter()
            .try_fold(size_of_val(&recursive), usize::checked_add)
            .and_then(|n| n.checked_mul(depth))?;
        let parts = [
            recursion,
            size_of::<Self>(),
            size_of::<SlicerStepRequirements>(),
            size_of::<SlicerStep<'_>>(),
            size_of::<SlicerStepFailure>(),
            size_of::<Cause>(),
            size_of::<Geometry>(),
            size_of::<&TokenizerSlice>(),
            size_of::<crate::Instant>(),
            size_of::<std::time::Duration>(),
            size_of::<TokenMaskConstructionRequirements>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, SlicerStepFailure>>(),
            size_of::<Result<SlicerStep<'_>, SlicerStepFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Option<SimpleVob>>(),
            size_of::<Vec<bool>>(),
            size_of::<(&mut SlicerStep<'_>, &mut ParserRecognizer<'_>, &[u8])>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Complete local geometry, derived from this source loan.
    pub fn requirements(&self) -> SlicerStepRequirements {
        self.requirements
    }
    /// Allocates each local destination once. The returned loan retains its tree;
    /// the enclosing source owner must retain the actual funding for these bytes.
    pub fn compile(self) -> Result<SlicerStep<'a>, SlicerStepFailure> {
        let mask = self.mask.compile().map_err(|e| rejected(Cause::Mask(e)))?;
        let mut flags = Vec::new();
        if let Err(e) = flags.try_reserve_exact(self.requirements.flags) {
            return Err(SlicerStepFailure {
                cause: Cause::Allocation(e),
                mask: Some(mask),
                flags,
            });
        }
        if flags.capacity() > self.requirements.flags {
            return Err(SlicerStepFailure {
                cause: Cause::Capacity,
                mask: Some(mask),
                flags,
            });
        }
        flags.resize(self.requirements.flags, false);
        Ok(SlicerStep {
            root: self.root,
            mask,
            flags,
        })
    }
}
/// Reusable local mask and traversal storage borrowing the exact immutable tree.
/// This object does not admit allocations performed by the supplied recognizer.
pub struct SlicerStep<'a> {
    root: &'a TokenizerSlice,
    mask: SimpleVob,
    flags: Vec<bool>,
}
impl fmt::Debug for SlicerStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicerStep")
            .field("match_flags", &self.flags.len())
            .finish_non_exhaustive()
    }
}
impl SlicerStep<'_> {
    /// Runs the ordinary slicer worker with no local vector growth. The borrowed
    /// mask is overwritten by the next invocation; parser/lexer work is separate.
    pub fn compute_bias(&mut self, rec: &mut ParserRecognizer<'_>, start: &[u8]) -> &SimpleVob {
        self.mask.set_all(false);
        self.root
            .compute_bias_into(rec, start, &mut self.mask, &mut self.flags);
        &self.mask
    }
    pub(super) fn into_mask(self) -> SimpleVob {
        self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::{GrammarInit, LLGuidanceOptions, NodeProps},
        earley::{BiasComputer, SlicedBiasComputer, SlicerConstructionPlan},
        grammar_builder::GrammarBuilder,
        toktrie::ApproximateTokEnv,
        ParserFactory, TokenParser,
    };
    use derivre::RegexAst;

    fn parser(factory: &ParserFactory, regex: &str) -> TokenParser {
        let mut builder = GrammarBuilder::new(
            None,
            factory.limits().clone(),
            derivre::ParserAllocationFunding::unenforced(),
        )
        .unwrap();
        builder
            .add_grammar(
                LLGuidanceOptions {
                    no_forcing: true,
                    ..LLGuidanceOptions::default()
                },
                RegexAst::NoMatch,
            )
            .unwrap();
        let rx = builder.regex.regex(regex).unwrap();
        let start = builder.lexeme_ext(rx, None, NodeProps::default()).unwrap();
        builder.set_start_node(start).unwrap();
        let mut parser = factory
            .create_parser_from_init_default(GrammarInit::Internal(
                builder.grammar,
                builder.regex.spec,
            ))
            .unwrap();
        parser.start_without_prompt();
        parser
    }

    #[test]
    fn reusable_slicer_step_matches_unsliced_parser_masks_and_retains_failed_mask() {
        let env = ApproximateTokEnv::single_byte_env();
        let patterns = vec![
            "[a-z]+".into(),
            "[a-m]+".into(),
            "[0-9]+".into(),
            "[A-Z]+".into(),
        ];
        let mut factory = ParserFactory::new(&env, Default::default(), &patterns).unwrap();
        factory.quiet();
        factory.limits_mut().precompute_large_lexemes = false;
        let ordinary = factory.slicer();
        let record = ordinary.source_plan().unwrap().compile().unwrap();
        let program = SlicerConstructionPlan::prepare(record.view(), env.tok_trie())
            .unwrap()
            .compile()
            .unwrap();
        let plan = program.step_plan().unwrap();
        assert_eq!(plan.requirements().match_flags(), 4); // three root children + nested a..m
        assert!(plan.requirements().required_bytes() > plan.requirements().buffer_bytes());
        let mut step = plan.compile().unwrap();
        let flags_ptr = step.flags.as_ptr();
        let flags_capacity = step.flags.capacity();
        let mask_ptr = step.mask.as_slice().as_ptr();
        // Multiple matching siblings, one nested match, and no slice match.
        for (regex, expected_count) in [("[a-z0-9]+", 36), ("[a-m]+", 13), ("@", 1)] {
            let mut ordinary_parser = parser(&factory, regex);
            let mut prepared_parser = parser(&factory, regex);
            let mut unsliced_parser = parser(&factory, regex);
            for start in [b"".as_slice(), b"a".as_slice(), b"".as_slice()] {
                let expected = ordinary_parser
                    .parser
                    .with_recognizer(|rec| ordinary.compute_bias(rec, start));
                let mut unsliced = env.tok_trie().alloc_token_set();
                unsliced_parser
                    .parser
                    .with_recognizer(|rec| env.tok_trie().add_bias(rec, &mut unsliced, start));
                prepared_parser.parser.with_recognizer(|rec| {
                    let actual = step.compute_bias(rec, start);
                    assert_eq!(actual, &expected);
                    assert_eq!(actual, &unsliced);
                    if start.is_empty() {
                        assert_eq!(actual.num_set(), expected_count);
                    }
                });
                assert_eq!(step.flags.as_ptr(), flags_ptr);
                assert_eq!(step.flags.capacity(), flags_capacity);
                assert_eq!(step.mask.as_slice().as_ptr(), mask_ptr);
            }
            if regex == "[a-z0-9]+" {
                assert!(prepared_parser.parser.stats().slices_applied >= 2);
            }
        }
        // Inject allocation refusal only after a real mask was constructed.
        // This is not an admissible source geometry or an original-parser claim.
        let mut fail = program.step_plan().unwrap();
        fail.requirements.flags = usize::MAX;
        let failed = fail.compile().unwrap_err();
        assert!(matches!(failed.cause, Cause::Allocation(_)));
        assert_eq!(
            failed.mask.as_ref().unwrap().as_slice().len(),
            step.mask.as_slice().len()
        );
        assert_eq!(failed.flags.capacity(), 0);
        drop(step);
        drop(program);
        drop(record);
        drop(ordinary);
        drop(factory);
        drop(env);
        assert!(!failed.mask.as_ref().unwrap().as_slice().is_empty());
        drop(failed);

        let empty_env = ApproximateTokEnv::single_byte_env();
        let no_slices = SlicedBiasComputer::new(&empty_env, &[]).unwrap();
        assert_eq!(
            SlicerStepPlan::prepare(&no_slices.top_slice, empty_env.tok_trie())
                .unwrap()
                .requirements()
                .match_flags(),
            0
        );
    }
}
