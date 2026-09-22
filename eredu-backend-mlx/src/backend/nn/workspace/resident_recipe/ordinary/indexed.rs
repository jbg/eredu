//! Indexed occurrences grouped by the existing model Work lifetime.
use super::super::*;
use crate::backend::{
    error::Error as NativeError,
    runtime::residency::parameter_bank::{
        OrdinaryIndexedOccurrence, OrdinaryIndexedRequestProgram,
    },
};
use eredu_runtime::working_memory::{InferenceTextStep, WorkingMemoryError};
use std::rc::Rc;

#[derive(Debug)]
struct PrefillProgram {
    spans: Vec<Rc<OrdinaryAddressableProgram>>,
    occurrences: usize,
}
impl OrdinaryIndexedRequestProgram for PrefillProgram {
    fn len(&self) -> usize {
        self.occurrences
    }
    fn occurrence(&self, mut index: usize) -> Option<OrdinaryIndexedOccurrence<'_>> {
        for span in &self.spans {
            if index < span.len() {
                return span.occurrence(index);
            }
            index = index.checked_sub(span.len())?;
        }
        None
    }
}

/// One provider spans all prefill chunks; every decode Work gets its own
/// ordered source program. These are descriptions, not execution authority.
pub(crate) struct OrdinaryIndexedPrograms {
    geometry: InferenceGeometry,
    prefill: Option<Rc<dyn OrdinaryIndexedRequestProgram>>,
    decode: Vec<Option<Rc<dyn OrdinaryIndexedRequestProgram>>>,
}
impl std::fmt::Debug for OrdinaryIndexedPrograms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryIndexedPrograms")
            .field("geometry", &self.geometry)
            .field(
                "prefill_occurrences",
                &self.prefill.as_ref().map(|p| p.len()),
            )
            .field("decode_steps", &self.decode.len())
            .finish()
    }
}
impl OrdinaryIndexedPrograms {
    /// Source identities stay borrowed from the cold planning owner. Cloning a
    /// shared handle here allocates no new program or source inventory.
    pub(crate) fn for_step(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<Option<Rc<dyn OrdinaryIndexedRequestProgram>>, NativeError> {
        let invalid = || NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch);
        if step.request().geometry() != self.geometry || prefill != (step.attempt() == 0) {
            return Err(invalid());
        }
        if prefill {
            return Ok(self.prefill.clone());
        }
        let index = step.attempt().checked_sub(1).ok_or_else(invalid)?;
        let index = usize::try_from(index).map_err(|_| invalid())?;
        self.decode.get(index).cloned().ok_or_else(invalid)
    }
    pub(crate) fn programs(&self) -> impl Iterator<Item = &dyn OrdinaryIndexedRequestProgram> {
        self.prefill
            .iter()
            .chain(self.decode.iter().filter_map(Option::as_ref))
            .map(|program| &**program)
    }
}

impl ResidentNativeRecipe {
    pub(crate) fn ordinary_indexed_programs(
        &self,
        context: &WorkspaceContext,
    ) -> Result<OrdinaryIndexedPrograms, Error> {
        let invalid =
            || context.metadata_error(format_args!("ordinary indexed request source mismatch"));
        let funding = context.metadata_funding().ok_or_else(invalid)?;
        if self
            .planning_metadata
            .as_ref()
            .is_none_or(|own| !own.same_account(&funding))
            || self.records.iter().any(|row| row.addressable.is_some())
        {
            return Err(invalid());
        }
        context.charge_metadata(size_of::<(
            &Self,
            &WorkspaceContext,
            OrdinaryIndexedPrograms,
            Result<OrdinaryIndexedPrograms, Error>,
            PrefillProgram,
        )>())?;
        let count = self
            .records
            .iter()
            .filter(|row| {
                matches!(row.span, InferenceWorkspaceSpan::Prefill(_))
                    && row.ordinary_addressable.is_some()
            })
            .count();
        let mut spans = context.metadata_vec(count)?;
        let mut occurrences = 0usize;
        let decode_count = self
            .records
            .iter()
            .filter(|row| matches!(row.span, InferenceWorkspaceSpan::Decode { .. }))
            .count();
        let mut decode = context.metadata_vec(decode_count)?;
        for row in &self.records {
            match &row.span {
                InferenceWorkspaceSpan::Prefill(_) => {
                    if !decode.is_empty() {
                        return Err(invalid());
                    }
                    if let Some(program) = &row.ordinary_addressable {
                        occurrences = occurrences
                            .checked_add(program.len())
                            .ok_or(WorkspaceMetadataError::Overflow)?;
                        spans.push(program.clone());
                    }
                }
                InferenceWorkspaceSpan::Decode { index, .. } => {
                    if usize::try_from(*index).ok() != Some(decode.len()) {
                        return Err(invalid());
                    }
                    decode.push(
                        row.ordinary_addressable.as_ref().map(|program| {
                            program.clone() as Rc<dyn OrdinaryIndexedRequestProgram>
                        }),
                    );
                }
                InferenceWorkspaceSpan::Sampling(_) => return Err(invalid()),
            }
        }
        let prefill = if occurrences == 0 {
            None
        } else {
            Some(context.metadata_rc(PrefillProgram { spans, occurrences })?
                as Rc<dyn OrdinaryIndexedRequestProgram>)
        };
        Ok(OrdinaryIndexedPrograms {
            geometry: self.plan.geometry(),
            prefill,
            decode,
        })
    }
}
