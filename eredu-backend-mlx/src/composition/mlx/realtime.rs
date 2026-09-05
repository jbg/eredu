//! MLX implementation of backend-neutral realtime loading and execution.

use std::{num::NonZeroUsize, ops::Deref, sync::Arc};

use eredu_architectures::moshi::{
    inspect_moshi_realtime, moshi_realtime_request_from_normalized,
    prepare_selected_moshi_realtime_source, select_inspected_moshi_realtime,
    MoshiRealtimeExecution, PreparedMoshiRealtimeSource, RealtimePreparationPlan,
};
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::cache::{StateComponentPolicy, StateComponentRole};
use eredu_core::{
    backend::Completion,
    realtime::{RealtimeDecisionDiagnostics, RealtimeInputFrame, RealtimeOutputFrame},
    CompletionCancellationMode, SessionCapabilities,
};
use eredu_runtime::{
    execute_realtime_frame, CommunicationCompletionCapabilities, CompletedRealtimeFrame,
    ExecutionResidency, GenerationSampler, MaterializedRealtimeInput,
    PreparedRealtimeFrameExecutor, PrepublicationRealtimeFrame, RealtimeArchitectureRequirements,
    RealtimeCompletionCreationError, RealtimeFrameCompletionMechanism, RealtimeFrameHostObserver,
    RealtimeFrameTensorMechanisms, RealtimeHostTokenMaterializer, RealtimeMechanism,
    RealtimeMechanismCapabilities, RealtimeObservationRequirements, RealtimePayloadBranch,
    RealtimePayloadHistory, RealtimeSessionBranch, StateComponentMechanism,
    StateComponentPlacement, StateMechanismCapabilities, WeightLoweringCapability,
    WeightLoweringKind,
};
use safemlx::{
    ops::{indexing::TryIndexOp, stack_axis},
    random,
    transforms::{async_eval_with_event, eval},
    Array, Dtype, Event, Stream,
};

use crate::backend::runtime::distributed::Group;
use crate::{
    backend::random::RandomState,
    backend::runtime::{
        cache::state::{MlxKeyValueState, MlxKeyValueTransactionBranch},
        generation::MlxSamplingBackend,
    },
    backend::{
        error::Error,
        nn::tensor::{TokenValidationBatch, TokenValidationScope},
    },
    composition::moshi::{self as neutral_moshi, MlxRealtimeExecution},
    MlxLoadRequest, MlxTensor,
};

mod completion;
mod frame_execution;
mod observation;
mod selection_loading;

pub use completion::MlxRealtimeCompletion;
pub use frame_execution::{MlxRealtimeFrameCompletionMechanism, MlxRealtimeFrameTensorMechanisms};
pub use observation::{MlxFrameSessionBranch, MlxPrepublicationFrame, MlxRealtimeHostObserver};
pub use selection_loading::MlxRealtimeExecutionContext;

#[cfg(test)]
use completion::{submit_or_synchronously_drain, CompletionSubmissionFailure};
use frame_execution::submit_scheduled_realtime_frame;
#[cfg(test)]
use selection_loading::{
    mlx_realtime_mechanisms, mlx_supports_realtime_lowering, realtime_session_capabilities,
    validate_realtime_session_requirements,
};

#[cfg(test)]
mod completion_ownership_tests;
#[cfg(test)]
pub(crate) mod tests;
