//! Architecture-declared token segments through the canonical numerical worker.
use super::*;
use crate::backend::OriginalCopyEnvironment;
use eredu_runtime::working_memory::WorkingMemoryError;
fn tensor(result: NumericalOutput) -> Result<OriginalNumericalValue, Error> {
    match result { NumericalOutput::Tensor(value) => Ok(value),
        _ => Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)) }
}
pub(crate) fn repeated_token_input(token: u32, positions: u64,
    sources: &OriginalSpeculativeNumericalSources, environment: &OriginalCopyEnvironment<'_>,
) -> Result<OriginalNumericalValue, Error> {
    let length = u32::try_from(positions).map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
    let (roots, mechanisms) = sources.numerical_prerequisites();
    tensor(NumericalProducer::execute_repeated_token(sources, environment, roots, mechanisms, token, length)?)
}
struct SequenceBody {
    values: Vec<OriginalNumericalValue>,
    _funding: HostMetadataFunding,
}
/// Same admitted ordered sources, shared with failure recovery until completion.
pub(super) struct TokenSequence(Option<Rc<SequenceBody>>);
impl Clone for TokenSequence {
    fn clone(&self) -> Self { Self(self.0.as_ref().map(Rc::clone)) }
}
impl Drop for TokenSequence {
    fn drop(&mut self) { if let Some(owner) = self.0.take() { drop(Rc::into_inner(owner)); } }
}
impl TokenSequence {
    pub(super) fn values(&self) -> &[OriginalNumericalValue] {
        &self.0.as_deref().expect("live token sequence").values
    }
}
pub(crate) fn concatenate_token_inputs(values: Vec<OriginalNumericalValue>,
    sources: &OriginalSpeculativeNumericalSources, environment: &OriginalCopyEnvironment<'_>,
) -> Result<OriginalNumericalValue, Error> {
    let funding = sources.metadata_funding();
    let controls = [rc_bytes::<SequenceBody>().ok_or_else(model::overflow)?,
        size_of::<TokenSequence>(), size_of::<Result<OriginalNumericalValue, Error>>(),
        size_of::<(Vec<OriginalNumericalValue>, &OriginalSpeculativeNumericalSources, &OriginalCopyEnvironment<'_>)>()];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or_else(model::overflow)?).map_err(Error::WorkspacePlanning)?;
    let sequence = TokenSequence(Some(Rc::new(SequenceBody { values, _funding: funding.clone() })));
    let (roots, mechanisms) = sources.numerical_prerequisites();
    let mut value = tensor(NumericalProducer::execute_token_concatenate(sources, environment,
        roots, mechanisms, &sequence)?)?;
    Rc::get_mut(value.0.as_mut().expect("unpublished token projection"))
        .expect("unique token projection")._readout_source = Some(RetainedValues::Sequence(sequence));
    Ok(value)
}
