//! MLX implementation of backend-neutral realtime loading and execution.

use std::{num::NonZeroUsize, ops::Deref, sync::Arc};

use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::cache::{StateComponentPolicy, StateComponentRole};
use eredu_core::{backend::Completion, realtime::RealtimeInputFrame, CompletionCancellationMode};
use eredu_runtime::{
    execute_realtime_frame, CommunicationCompletionCapabilities, CompletedRealtimeFrame,
    ExecutionResidency, GenerationSampler, PreparedRealtimeFrameExecutor,
    PrepublicationRealtimeFrame, RealtimeCompletionCreationError, RealtimeCompletionMechanism,
    RealtimeFrameHostObserver, RealtimeFramePlan, RealtimeFrameResources,
    RealtimeFrameTensorMechanisms, RealtimeHostOutputProjection, RealtimeHostTokenMaterializer,
    RealtimeMechanism, RealtimePayloadBranch, RealtimeSessionBranch, StateComponentPlacement,
    WeightLoweringKind,
};
#[cfg(test)]
use safemlx::ops::{indexing::TryIndexOp, stack_axis};
use safemlx::{transforms::async_eval_with_event, Array, Event, Stream};

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
    MlxTensor,
};

mod completion;
pub(crate) mod construction;
mod execution;
mod frame_execution;
mod observation;
mod selection_loading;
mod startup;
pub(crate) mod submission;

pub use completion::MlxRealtimeCompletion;
pub(crate) use completion::MlxRealtimeExecutionResources;
pub use execution::MlxRealtimeExecution;
pub use observation::{MlxFrameSessionBranch, MlxPrepublicationFrame};
pub use selection_loading::{MlxRealtimeBinding, MlxRealtimeExecutionContext, MlxRealtimeSupport};
pub use submission::SharedRealtimeExecution;

use frame_execution::submit_scheduled_realtime_frame;

#[cfg(test)]
mod completion_ownership_tests;
#[cfg(test)]
pub(crate) mod tests;
