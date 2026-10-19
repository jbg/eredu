//! MLX implementation of backend-neutral realtime loading and execution.

use std::{num::NonZeroUsize, ops::Deref, sync::Arc};

use eredu_architectures::moshi::{
    inspect_moshi_realtime, moshi_realtime_request_from_normalized,
    prepare_selected_moshi_realtime_source, select_inspected_moshi_realtime, MoshiPreparationPlan,
    MoshiRealtimeExecution, PreparedMoshiRealtimeSource,
};
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::cache::{StateComponentPolicy, StateComponentRole};
use eredu_core::{
    backend::Completion,
    realtime::{RealtimeInputFrame, RealtimeOutputFrame},
    CompletionCancellationMode, SessionCapabilities,
};
use eredu_runtime::{
    execute_realtime_frame, CommunicationCompletionCapabilities, CompletedRealtimeFrame,
    ExecutionResidency, GenerationSampler, PreparedRealtimeFrameExecutor,
    PrepublicationRealtimeFrame, RealtimeArchitectureRequirements, RealtimeCompletionCreationError,
    RealtimeCompletionMechanism, RealtimeFrameHostObserver, RealtimeFrameResources,
    RealtimeFrameTensorMechanisms, RealtimeHostTokenMaterializer, RealtimeMechanism,
    RealtimeMechanismCapabilities, RealtimeObservationRequirements, RealtimePayloadBranch,
    RealtimeSessionBranch, StateComponentPlacement, WeightLoweringKind,
};
#[cfg(test)]
use safemlx::ops::{indexing::TryIndexOp, stack_axis};
use safemlx::{random, transforms::async_eval_with_event, Array, Event, Stream};

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
pub(crate) use completion::{MlxRealtimeExecutionResources, MlxRealtimeSubmissionScope};
pub use frame_execution::{MlxRealtimeCompletionMechanism, MlxRealtimeFrameTensorMechanisms};
pub use observation::{MlxFrameSessionBranch, MlxPrepublicationFrame, MlxRealtimeHostObserver};
pub use selection_loading::{MlxPreparedRealtimeExecution, MlxRealtimeExecutionContext};

use frame_execution::submit_scheduled_realtime_frame;

#[cfg(test)]
mod completion_ownership_tests;
#[cfg(test)]
pub(crate) mod tests;
