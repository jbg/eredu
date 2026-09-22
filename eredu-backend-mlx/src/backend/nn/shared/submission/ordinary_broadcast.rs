//! Ordinary publication retains the input, rank contribution and returned root.
use super::*;
use crate::backend::error::Error;
use crate::backend::nn::workspace::OrdinaryCallControls;
use crate::backend::runtime::distributed::completion::prepared::CompletionResourceLayout;
use std::mem::{size_of, size_of_val};

const RETAINED_ARRAYS: usize = 3;

/// The same fixed population is retained by the ordinary publication worker and
/// quoted for its completion. The caller supplies its existing output alias.
pub(super) fn retained_arrays(input: Array, contribution: Array, output: Array) -> Vec<Array> {
    let retained: [Array; RETAINED_ARRAYS] = [input, contribution, output];
    retained.into()
}

impl MlxNeuralBackend {
    /// Caller/completion controls of ordinary Broadcast. Its rank contribution
    /// is already emitted by the shared numerical trace, and the exact retained
    /// group supplies its native Sum population independently.
    pub(crate) fn ordinary_broadcast_call_controls(
        group: &Group,
        root: usize,
    ) -> Option<OrdinaryCallControls> {
        if root >= group.size() || group.rank() >= group.size() {
            return None;
        }
        let mut controls = MlxCommunicationCompletion::ordinary_submit_call_controls(
            1,
            CompletionResourceLayout {
                arrays: RETAINED_ARRAYS,
                counts: &[],
                groups: 1,
                routes: 0,
                streams: 1,
            },
        )?;
        let parts = [
            group.retention_copy_bytes()?,
            size_of::<[Array; RETAINED_ARRAYS]>(),
            size_of::<Vec<Array>>(),
            size_of::<MlxTensor>(),
            size_of::<Submission<MlxTensor, MlxNeuralCommunicationCompletion>>(),
            size_of::<Result<Submission<MlxTensor, MlxNeuralCommunicationCompletion>, Error>>(),
            size_of::<(&Group, &Stream, usize)>(),
            size_of::<(&Array, MlxCommunicationDtypes)>(),
            size_of::<Option<safemlx::RuntimeCallGuard>>().checked_mul(2)?,
            size_of::<Result<Option<safemlx::RuntimeCallGuard>, safemlx::error::Exception>>()
                .checked_mul(2)?,
            size_of::<Result<(), safemlx::error::Exception>>(),
            size_of::<Result<Array, safemlx::error::Exception>>(),
            size_of::<(Array, Array, Array)>(),
            size_of::<Result<MlxCommunicationCompletion, safemlx::error::Exception>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?;
        controls.metadata_bytes = controls
            .metadata_bytes
            .checked_add(u64::try_from(bytes).ok()?)?;
        Some(controls)
    }
}
