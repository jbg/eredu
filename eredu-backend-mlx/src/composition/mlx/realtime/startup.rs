//! MLX state and RNG mechanisms for portable realtime construction.

use eredu_core::BackendFailure;
use eredu_runtime::{
    CacheResidencyPolicy, RealtimeCopyLimits, RealtimeStartupMechanisms, RealtimeStateGroup,
    RealtimeStateMaterializer, SelectedRealtimeStateRealization, SelectedStateRealization,
};

use super::{MlxRealtimeCompletion, MlxRealtimeExecutionContext};
use crate::backend::{
    random::RandomState,
    runtime::cache::state::{MlxHybridState, MlxJointState, MlxKeyValueState},
};
use crate::composition::mlx::replicated_text::MlxStateMechanisms;

impl RealtimeStartupMechanisms for MlxRealtimeExecutionContext {
    type Random = RandomState;
    type Completion = MlxRealtimeCompletion;

    fn realize_random(&self, seed: u64) -> Result<RandomState, BackendFailure> {
        RandomState::with_seed(seed).map_err(BackendFailure::from_error)
    }
}

impl<S: MlxStateMechanisms> RealtimeStateMaterializer<SelectedStateRealization, S>
    for MlxRealtimeExecutionContext
{
    fn realize_state(&self, selected: &SelectedStateRealization) -> Result<S, BackendFailure> {
        S::realize(selected, None, 0).map_err(BackendFailure::from_error)
    }
}

impl RealtimeStateMaterializer<SelectedRealtimeStateRealization, MlxKeyValueState>
    for MlxRealtimeExecutionContext
{
    fn realize_state(
        &self,
        selected: &SelectedRealtimeStateRealization,
    ) -> Result<MlxKeyValueState, BackendFailure> {
        if !matches!(selected.policy(), CacheResidencyPolicy::Device) {
            return Err(BackendFailure::from_error(
                crate::backend::error::Error::Parallel(
                    "MLX realtime key/value state requires device residency".into(),
                ),
            ));
        }
        MlxKeyValueState::device(selected.layout().clone()).map_err(BackendFailure::from_error)
    }
}

impl RealtimeStateGroup<MlxHybridState, MlxKeyValueState> for MlxRealtimeExecutionContext {
    type State = MlxJointState;

    fn group_state(
        &self,
        left: MlxHybridState,
        right: MlxKeyValueState,
        limits: RealtimeCopyLimits,
    ) -> Result<Self::State, BackendFailure> {
        MlxJointState::new(left, right, self.stream(), limits).map_err(BackendFailure::from_error)
    }
}
