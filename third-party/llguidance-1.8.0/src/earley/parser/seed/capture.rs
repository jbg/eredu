//! Paid definitive capture bytes beside the source-named capture records.
use super::super::{agenda, capture, CSymIdx, Lexeme};
use super::{agenda::InitialCapture, reserve, Cause, PreparedEarleySeed};
use std::mem::{size_of, size_of_val};
use toktrie::{RawTokenDecodeError, RawTokenDecodeFailure, RawTokenDecodePlan, TokTrie};
impl PreparedEarleySeed {
    pub(super) fn capture_source<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        trie: &TokTrie,
        symbol: CSymIdx,
        stop: bool,
        start: usize,
        row: usize,
        lexeme: &Lexeme,
        is_lexeme: bool,
        funding: &F,
    ) -> Result<(), Cause<E>> {
        let parts = [
            RawTokenDecodePlan::inspection_control_bytes().ok_or(Cause::Overflow)?,
            size_of::<RawTokenDecodePlan<'_>>(),
            size_of::<RawTokenDecodeFailure>(),
            size_of::<RawTokenDecodeError>(),
            size_of::<InitialCapture>(),
            size_of::<Cause<E>>(),
            size_of::<(&mut Self, &TokTrie, &Lexeme, &F)>(),
            size_of::<(&mut Vec<u8>, &F)>(),
            size_of::<(CSymIdx, bool, usize, usize, bool)>(),
            size_of::<Result<Vec<u8>, RawTokenDecodeFailure>>(),
            size_of::<Result<RawTokenDecodePlan<'_>, RawTokenDecodeError>>(),
            size_of::<Result<(), Cause<E>>>(),
            size_of::<Result<(), E>>(),
            size_of::<std::slice::Iter<'_, super::super::RowInfo>>(),
            size_of::<std::iter::Rev<std::slice::Iter<'_, InitialCapture>>>(),
            size_of::<Option<&[u8]>>(),
        ];
        funding(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Cause::Overflow)?,
        )
        .map_err(Cause::Funding)?;
        if start > row || row > self.row_infos.len() || self.capture_pending.is_some() {
            return Err(Cause::Source);
        }
        self.capture_raw.clear();
        let raw = &mut self.capture_raw;
        capture::visit(
            &self.row_infos,
            start,
            row,
            lexeme,
            is_lexeme,
            stop,
            |part| {
                let total = raw.len().checked_add(part.len()).ok_or(Cause::Overflow)?;
                reserve(raw, total, funding)?;
                raw.extend_from_slice(part);
                Ok(())
            },
        )?;
        let plan = trie.raw_decode_plan(raw).map_err(Cause::DecodeSource)?;
        funding(plan.required_bytes()).map_err(Cause::Funding)?;
        self.capture_pending = Some(plan.compile().map_err(Cause::Decode)?);
        let record = InitialCapture {
            symbol,
            stop,
            bytes: Vec::new(),
        };
        let name = record.name(&self.grammar);
        let previous = self
            .captures
            .iter()
            .rev()
            .find(|entry| entry.name(&self.grammar) == name)
            .map(|entry| entry.bytes.as_slice());
        let bytes = self.capture_pending.as_ref().expect("decoded capture");
        if agenda::capture_changed(name, bytes, previous) {
            let count = self.captures.len().checked_add(1).ok_or(Cause::Overflow)?;
            reserve(&mut self.captures, count, funding)?;
            self.captures.push(InitialCapture {
                bytes: self.capture_pending.take().expect("capture bytes"),
                ..record
            });
        } else {
            self.capture_pending = None;
        }
        self.capture_raw.clear();
        Ok(())
    }
}
