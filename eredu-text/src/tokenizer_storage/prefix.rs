//! Fresh input projection sharing the original immutable tokenizer residence.
use super::*;
use tokenizers::{AddedVocabulary, AddedVocabularyCompileFailure, AddedVocabularyRefreshPlan};

#[derive(Debug)]
pub(super) struct Projection {
    pub(super) added: Option<AddedVocabulary>,
    pub(super) decoder: Option<PreparedDecodeSource>,
    failure: Option<Cause>,
}
/// One attempt over the same immutable root; no JSON reconstruction or model copy.
#[derive(Debug)]
pub struct InputPrefixPlan<'a> {
    source: &'a PreparedTokenizer,
    added: Option<AddedVocabularyRefreshPlan<'a>>,
    requirements: TokenizerRequirements,
}
impl PreparedTokenizer {
    /// Inspect the actual source projection before admission of its destinations.
    /// Identity removal returns no plan and requires no new source residence.
    pub fn input_prefix_plan(&self) -> Result<Option<InputPrefixPlan<'_>>, TokenizerSourceError> {
        if self.input_prefix_removal_is_identity() {
            return Ok(None);
        }
        let added = AddedVocabularyRefreshPlan::for_input_prefix_removal(&self.root().tokenizer)
            .map_err(|e| TokenizerSourceError::Root(TokenizerCompileError::Added(e)))?;
        let (buffers, controls) = if let Some(added) = &added {
            let a = added.requirements();
            (
                a.buffer_bytes()
                    .checked_add(self.root().envelope.buffer_bytes())
                    .ok_or_else(TokenizerSourceError::overflow)?,
                a.control_bytes()
                    .checked_add(self.root().envelope.control_bytes())
                    .ok_or_else(TokenizerSourceError::overflow)?,
            )
        } else {
            (0, 0)
        };
        // One concrete destination exists before refresh starts. Success and
        // failure retain that same owner, so returning a failure never boxes
        // an already-large partial aggregate or allocates another error shell.
        let buffers = buffers
            .checked_add(size_of::<Projection>())
            .ok_or_else(TokenizerSourceError::overflow)?;
        let controls = [
            controls,
            size_of::<InputPrefixPlan<'_>>(),
            size_of::<Result<Option<InputPrefixPlan<'_>>, TokenizerSourceError>>(),
            size_of::<InputPrefixFailure>(),
            size_of::<Cause>(),
            size_of::<Projection>(),
            size_of::<Box<Projection>>(),
            size_of::<Option<Box<Projection>>>(),
            size_of::<PreparedTokenizer>(),
            size_of::<Option<AddedVocabulary>>(),
            size_of::<Option<PreparedDecodeSource>>(),
            size_of::<Arc<Root>>(),
            size_of::<Result<PreparedTokenizer, InputPrefixFailure>>(),
            size_of::<Result<DecodeCompilePlan<'_>, DecodeSourceError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(TokenizerSourceError::overflow)?;
        let requirements = TokenizerRequirements {
            buffers,
            controls,
            total: buffers
                .checked_add(controls)
                .ok_or_else(TokenizerSourceError::overflow)?,
        };
        Ok(Some(InputPrefixPlan {
            source: self,
            added,
            requirements,
        }))
    }
}
impl InputPrefixPlan<'_> {
    /// Fresh changed-table/decoder buffers and named constructor controls.
    pub fn requirements(&self) -> TokenizerRequirements {
        self.requirements
    }
    /// Construct once. Root aliases allocate no new shell; the original root's
    /// account must remain alive through this derivative's final retirement.
    pub fn compile(self) -> Result<PreparedTokenizer, InputPrefixFailure> {
        let root = Arc::clone(self.source.root.as_ref().expect("live root"));
        let mut projection = Box::new(Projection {
            added: None,
            decoder: None,
            failure: None,
        });
        let result = (|| -> Result<Option<PreparedDecodeSource>, Cause> {
            let Some(plan) = self.added else {
                return Ok(None);
            };
            projection.added = Some(plan.compile().map_err(Cause::Added)?);
            let view = root
                .tokenizer
                .input_prefix_view(projection.added.as_ref().expect("constructed projection"));
            let plan = DecodeCompilePlan::prepare_input(view).map_err(Cause::Source)?;
            let actual = plan.requirements();
            if actual.id_slots() > root.envelope.id_slots()
                || actual.piece_bytes() > root.envelope.piece_bytes()
            {
                return Err(Cause::Source(DecodeSourceError::Overflow));
            }
            Ok(Some(plan.compile().map_err(Cause::Decode)?))
        })();
        match result {
            Ok(decoder) => {
                projection.decoder = decoder;
                Ok(PreparedTokenizer {
                    projection: Some(projection),
                    root: Some(root),
                })
            }
            Err(cause) => {
                projection.failure = Some(cause);
                Err(InputPrefixFailure {
                    projection: Some(projection),
                    root: Some(root),
                })
            }
        }
    }
}
#[derive(Debug)]
enum Cause {
    Source(DecodeSourceError),
    Added(AddedVocabularyCompileFailure),
    Decode(DecodeCompileFailure),
}
/// The actual failed prefix and original shared source, without retry/extraction.
#[derive(Debug)]
pub struct InputPrefixFailure {
    projection: Option<Box<Projection>>,
    root: Option<Arc<Root>>,
}
impl Drop for InputPrefixFailure {
    fn drop(&mut self) {
        drop(self.projection.take());
        if let Some(root) = self.root.take() {
            drop(Arc::into_inner(root));
        }
    }
}
impl fmt::Display for InputPrefixFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cause() {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Added(e) => fmt::Display::fmt(e, f),
            Cause::Decode(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for InputPrefixFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self.cause() {
            Cause::Source(e) => e,
            Cause::Added(e) => e,
            Cause::Decode(e) => e,
        })
    }
}
impl InputPrefixFailure {
    fn cause(&self) -> &Cause {
        self.projection
            .as_ref()
            .expect("live failed projection")
            .failure
            .as_ref()
            .expect("failed projection cause")
    }
}
