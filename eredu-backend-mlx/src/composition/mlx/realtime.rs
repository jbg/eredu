//! MLX implementation of backend-neutral realtime loading and execution.

use eredu_core::realtime_mechanisms::{
    RealtimeCompletionCreationError, RealtimeFrameTensorMechanisms, RealtimeHostTokenMaterializer,
};
use std::{num::NonZeroUsize, sync::Arc};

use eredu_checkpoint::lowering::WeightLoweringDescriptor;
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::cache::{StateComponentPolicy, StateComponentRole};
use eredu_core::communication::CommunicationCompletionCapabilities;
use eredu_core::{backend::Completion, CompletionCancellationMode};
use eredu_core::{
    preparation::{StateComponentPlacement, StateLifecycleCapabilities, WeightResidencyMechanism},
    realtime_mechanisms::{
        RealtimeLoadSupport, RealtimeMechanism, RealtimeMechanismFacts, RealtimeMechanismSupport,
    },
};
use safemlx::{transforms::async_eval_with_event, Array, Event, Stream};

use crate::backend::runtime::distributed::Group;
use crate::{
    backend::runtime::generation::MlxSamplingBackend,
    backend::{
        error::Error,
        nn::tensor::{TokenValidationBatch, TokenValidationScope},
    },
    MlxTensor,
};

mod completion;
mod frame_execution;
mod mechanisms;
mod selection_loading;
mod startup;
pub(crate) mod submission;

pub use completion::MlxRealtimeCompletion;
pub use selection_loading::{MlxRealtimeBinding, MlxRealtimeExecutionContext, MlxRealtimeSupport};
pub use submission::MlxRealtimeQueue;
