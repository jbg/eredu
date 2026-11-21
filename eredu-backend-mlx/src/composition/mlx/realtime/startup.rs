//! MLX random-state and completion mechanisms for portable realtime startup.

use eredu_core::realtime_mechanisms::RealtimeStartupMechanisms;
use eredu_core::BackendFailure;

use super::{MlxRealtimeCompletion, MlxRealtimeExecutionContext};
use crate::backend::random::RandomState;

impl RealtimeStartupMechanisms for MlxRealtimeExecutionContext {
    type Random = RandomState;
    type Completion = MlxRealtimeCompletion;

    fn realize_random(&self, seed: u64) -> Result<RandomState, BackendFailure> {
        RandomState::with_seed(seed).map_err(BackendFailure::from_error)
    }
}
