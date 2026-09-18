use super::*;
use regex_automata::util::source_storage as storage;

impl Prog {
    pub(crate) fn visit_source_storage(
        &self,
        visitor: &mut dyn storage::Visitor,
    ) -> core::result::Result<(), storage::Error> {
        if let Some(pool) = &self.scratch_pool {
            pool.visit_source_storage(visitor, |_, _| Ok(()))?;
        }
        storage::string(&self.seek_pattern, visitor);
        if !storage::vector(&self.body, visitor) {
            return Ok(());
        }
        for instruction in &self.body {
            match instruction {
                Insn::Lit(text) => {
                    storage::string(text, visitor);
                }
                Insn::CharClass(CharClassMatcher::Byte(ranges)) => {
                    storage::boxed(ranges, visitor);
                }
                Insn::CharClass(CharClassMatcher::Codepoint(ranges)) => {
                    storage::boxed(ranges, visitor);
                }
                Insn::Delegate(delegate) | Insn::AbsentRepeater(delegate) => {
                    storage::string(&delegate.pattern, visitor);
                    delegate.inner.visit_source_storage(visitor)?;
                }
                Insn::PikeDelegate(delegate) => {
                    delegate.inner.visit_source_storage(visitor)?;
                }
                Insn::DfaDelegate(delegate) => {
                    if storage::boxed(delegate, visitor) {
                        delegate.visit_source_storage(visitor)?;
                    }
                }
                Insn::Seek(seek) => {
                    storage::string(&seek.pattern, visitor);
                    seek.inner.visit_source_storage(visitor)?;
                }
                #[cfg(feature = "variable-lookbehinds")]
                Insn::BackwardsDelegate(delegate) => {
                    storage::string(&delegate.pattern, visitor);
                    delegate
                        .cache_pool
                        .visit_source_storage(visitor, |create, visitor| {
                            storage::boxed(create, visitor);
                            Ok(())
                        })?;
                    if storage::arc(&delegate.dfa, 0, visitor)? {
                        delegate.dfa.visit_source_storage(visitor)?;
                    }
                    if let Some(regex) = &delegate.capture_group_extraction_inner {
                        regex.visit_source_storage(visitor)?;
                    }
                }
                Insn::End
                | Insn::Any
                | Insn::AnyNoNL
                | Insn::AnyNoCRLF
                | Insn::Assertion(_)
                | Insn::Split(_, _)
                | Insn::SplitUnanchored(_, _)
                | Insn::Jmp(_)
                | Insn::Save(_)
                | Insn::Save0(_)
                | Insn::SaveCaptureGroupStart(_)
                | Insn::Restore(_)
                | Insn::RepeatGr { .. }
                | Insn::RepeatNg { .. }
                | Insn::RepeatEpsilonGr { .. }
                | Insn::RepeatEpsilonNg { .. }
                | Insn::FailNegativeLookAround
                | Insn::GoBack(_)
                | Insn::Backref { .. }
                | Insn::BeginAtomic
                | Insn::EndAtomic
                | Insn::ContinueFromPreviousMatchEnd { .. }
                | Insn::BackrefExistsCondition(_)
                | Insn::Fail
                | Insn::RejectEmptyMatchAtEOFFollowingNewline => {}
            }
        }
        Ok(())
    }
}
