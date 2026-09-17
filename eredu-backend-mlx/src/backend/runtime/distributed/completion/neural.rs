//! Typed model-communication errors over the existing native completion owner.
use super::{MlxCommunicationCompletion,OriginalCommunicationCompletion};
use crate::backend::error::Error;

/// Internal neural-backend completion adapter. The same native event, resource
/// owner, timeout and quarantine worker remain authoritative; this adapter only
/// preserves backend errors alongside their original native source.
#[derive(Debug)]
#[must_use = "retain or complete submitted communication"]
pub struct MlxNeuralCommunicationCompletion(MlxCommunicationCompletion);
impl MlxNeuralCommunicationCompletion {
    /// Move the same native owner into an existing native adapter contract.
    pub(crate) fn into_native(self)->MlxCommunicationCompletion {self.0}
    /// Preserve the existing ordinary scalar result consumer and its owner.
    pub(crate) fn with_f32_flag(self,output:safemlx::Array)
        ->(super::communication::MlxCommunicationFlag,Self) {
        let (flag,completion)=self.0.with_f32_flag(output);
        (flag,Self(completion))
    }
    #[cfg(test)]
    pub(crate) fn retained_arrays(&self)->usize {self.0.retained_arrays()}
    #[cfg(test)]
    pub(crate) fn retained_count_buffers(&self)->usize {self.0.retained_count_buffers()}
    #[cfg(test)]
    pub(crate) fn retained_groups(&self)->usize {self.0.retained_groups()}
    #[cfg(test)]
    pub(crate) fn retained_routes(&self)->usize {self.0.retained_routes()}
    #[cfg(test)]
    pub(crate) fn retained_streams(&self)->usize {self.0.retained_streams()}
    #[cfg(test)]
    pub(crate) fn submitted_outputs(&self)->usize {self.0.submitted_outputs()}
}
impl From<MlxCommunicationCompletion> for MlxNeuralCommunicationCompletion {
    fn from(value:MlxCommunicationCompletion)->Self {Self(value)}
}
impl From<OriginalCommunicationCompletion> for MlxNeuralCommunicationCompletion {
    fn from(value:OriginalCommunicationCompletion)->Self {Self(value.0)}
}
impl eredu_core::Completion for MlxNeuralCommunicationCompletion {
    type Error=Error;
    fn resources_releasable(&self)->bool {eredu_core::Completion::resources_releasable(&self.0)}
    fn is_complete(&self)->Result<bool,Error> {eredu_core::Completion::is_complete(&self.0).map_err(Error::from)}
    fn wait(&self)->Result<(),Error> {eredu_core::Completion::wait(&self.0).map_err(Error::from)}
}
impl eredu_core::BoundedCompletion for MlxNeuralCommunicationCompletion {
    fn wait_bounded(self,policy:eredu_core::BoundedCompletionWait)
        ->Result<eredu_core::BoundedCompletionOutcome,Error> {
        eredu_core::BoundedCompletion::wait_bounded(self.0,policy).map_err(Error::from)
    }
}

/// Fixed move/error/return frames of the model adapter. There is no new Rc,
/// native operation, error allocation or authority held by the wrapper itself.
pub(super) fn control_bytes()->Option<usize> {
    use std::mem::{size_of,size_of_val};
    let frames=[size_of::<MlxNeuralCommunicationCompletion>(),
        size_of::<MlxCommunicationCompletion>(),size_of::<OriginalCommunicationCompletion>(),
        size_of::<Error>(),size_of::<Result<bool,Error>>(),size_of::<Result<(),Error>>(),
        size_of::<Result<eredu_core::BoundedCompletionOutcome,Error>>(),
        size_of::<(&MlxNeuralCommunicationCompletion,eredu_core::BoundedCompletionWait)>(),
        size_of::<Result<MlxNeuralCommunicationCompletion,Error>>()];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
