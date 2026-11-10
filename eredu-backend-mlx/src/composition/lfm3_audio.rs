//! MLX binding for the architecture-owned resident Audio driver.

use crate::backend::{
    nn::shared::MlxNeuralBackend,
    random::RandomState,
    runtime::{
        cache::state::MlxHybridState,
        checkpoint::binding::{
            build_neutral_module_bindings_with_recipes, materialize_module_bindings,
            populate_module_from_arrays_excluding,
        },
        generation::MlxSamplingBackend,
    },
    ExecutionContext,
};
use crate::{MlxTensor, Stream};
use eredu_architectures::lfm3::audio::{
    AudioAdvance, AudioPhase, AudioRequest, AudioSession, AudioSessionError, BindingSource,
    PreparedAudio, ResidentAudio, ResidentAudioBinding, ResidentAudioError,
};
use eredu_core::BackendFailure;
use eredu_nn::Parameterized;
use eredu_runtime::{RuntimeLayerState, StateLayout};

struct Binding {
    execution: ExecutionContext,
    weights: ExecutionContext,
}

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

impl ResidentAudioBinding<MlxNeuralBackend> for Binding {
    type State = MlxHybridState;
    type SamplingError = safemlx::error::Exception;
    type Sampling = MlxSamplingBackend;

    fn context(&self) -> &Stream {
        self.execution.stream()
    }

    fn bind<M: Parameterized<MlxTensor>>(
        &self,
        module: &mut M,
        weights: &BindingSource,
    ) -> Result<(), BackendFailure> {
        let source = weights.source();
        let mut recipes = weights.recipes().clone();
        let bindings =
            build_neutral_module_bindings_with_recipes(module, source.as_ref(), &mut recipes)
                .map_err(failure)?;
        let arrays = materialize_module_bindings(
            source.as_ref(),
            &bindings,
            self.weights.stream(),
            self.context(),
        )
        .map_err(failure)?;
        populate_module_from_arrays_excluding(module, &arrays, |_| false).map_err(failure)
    }

    fn state(&self, layout: &StateLayout) -> Result<Self::State, BackendFailure> {
        MlxHybridState::device(layout.clone(), &[]).map_err(failure)
    }

    fn branch_state(state: &Self::State) -> Result<Self::State, BackendFailure> {
        // This binding constructs only device-local state. Append roots are immutable.
        state.deep_clone_state().map_err(failure)
    }

    fn retained_values(state: &Self::State) -> Vec<&MlxTensor> {
        state
            .layers()
            .iter()
            .flat_map(RuntimeLayerState::retained_values)
            .collect()
    }
}

type Model = ResidentAudio<MlxNeuralBackend, Binding>;

/// Bound resident LFM3 Audio graph using exact F32 widening of floating source weights.
pub struct MlxAudioModel(Model);

impl MlxAudioModel {
    /// Binds an admitted Audio artifact on an explicitly selected device.
    /// The weights context is a separate host-side materialization stream.
    pub fn load(
        artifact: PreparedAudio,
        execution: ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        let weights = ExecutionContext::cpu().map_err(failure)?;
        Ok(Self(ResidentAudio::new(
            artifact,
            Binding { execution, weights },
        )?))
    }

    /// Prefills the request and creates a controlled session with an explicit seed.
    pub fn start(
        self,
        request: AudioRequest<'_>,
        seed: u64,
    ) -> Result<MlxAudioSession, ResidentAudioError> {
        let random = RandomState::with_seed(seed).map_err(failure)?;
        Ok(MlxAudioSession(self.0.start(request, random)?))
    }
}

/// Batch-one resident speech session exposing portable committed events.
pub struct MlxAudioSession(AudioSession<Model>);

impl MlxAudioSession {
    /// Executes a single text decision or depth pass.
    pub fn advance(&mut self) -> Result<AudioAdvance, AudioSessionError> {
        self.0.advance()
    }

    /// Drains the same controlled driver, omitting intermediate depth progress.
    pub fn events(&mut self) -> impl Iterator<Item = Result<AudioAdvance, AudioSessionError>> + '_ {
        self.0.events()
    }

    /// Current phase of the session.
    pub fn phase(&self) -> AudioPhase {
        self.0.phase()
    }

    /// Number of committed language slots.
    pub fn committed_slots(&self) -> usize {
        self.0.committed_slots()
    }

    /// Cancels at the next idle boundary.
    pub fn cancel(&mut self) {
        self.0.cancel();
    }
}
