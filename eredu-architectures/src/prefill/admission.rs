//! The existing admission owners lent into one shared prepared-input worker.
use super::*;
use crate::media_plan::{AdmittedCompositeInput, BoundPreparedMediaSemantics};
use eredu_runtime::input::PreparedModelInputOwner;

pub(crate) enum PrefillAdmission<P> {
    Ordinary(AdmittedCompositeInput<P>),
    Original(BoundPreparedMediaSemantics),
}
impl<P> PrefillAdmission<P> {
    pub(crate) fn input<'a, T>(
        &'a self,
        prepared: &'a PreparedModelInputOwner<T>,
        context: Option<&'a WorkspaceContext>,
    ) -> Result<PreparedCompositeInput<'a, T, P>, Error> {
        let metadata = Metadata::new(context);
        metadata.controls::<(
            PreparedCompositeInput<'a, T, P>,
            Result<PreparedCompositeInput<'a, T, P>, Error>,
        )>()?;
        let input = match self {
            Self::Ordinary(admitted) => {
                PreparedCompositeInput::new_with_diagnostic(prepared, admitted, |message| {
                    metadata.error(format_args!("{message}"))
                })?
            }
            Self::Original(original) => {
                if !prepared
                    .original_source()
                    .is_some_and(|source| source.same_source(original.source()))
                {
                    return Err(metadata.error(format_args!(
                        "compiled prefill admission differs from its original input owner"
                    )));
                }
                PreparedCompositeInput::from_original_with_diagnostic(
                    prepared,
                    original,
                    |message| metadata.error(format_args!("{message}")),
                )?
            }
        };
        Ok(input.with_metadata_loan(context))
    }
    // Only called after the funded constructor validated this unchanged pair.
    pub(crate) fn loan<'a, T>(
        &'a self,
        prepared: &'a PreparedModelInputOwner<T>,
        context: Option<&'a WorkspaceContext>,
    ) -> PreparedCompositeInput<'a, T, P> {
        let input = match self {
            Self::Ordinary(admitted) => {
                PreparedCompositeInput::new_with_diagnostic(prepared, admitted, |_| ())
            }
            Self::Original(original) => {
                PreparedCompositeInput::from_original_with_diagnostic(prepared, original, |_| ())
            }
        };
        input
            .expect("closed source retains validated admission pair")
            .with_metadata_loan(context)
    }
}
