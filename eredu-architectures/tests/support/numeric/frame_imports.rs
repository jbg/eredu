use super::*;
use eredu_core::{
    scheduler::{
        RequestId, RequestStatus, SchedulerError, SchedulerLimits, SemanticStateTransaction,
    },
    RealtimeFrameScheduleState, RealtimeInputFrame, RealtimeSampling, RealtimeSlotCoordinate,
    SessionCapabilities,
};
use eredu_runtime::{
    construct_realtime_model, ActivationObserver, CacheResidencyPolicy,
    CommunicationCompletionCapabilities, CompletedRealtimeFrame, ExecutionResidency,
    MaterializedRealtimeInput, NormalizedLoadRequest, RealizedRealtimePolicy,
    RealizedRealtimeState, RealtimeArchitectureIdentity, RealtimeCompletionCreationError,
    RealtimeFrameCompletionMechanism, RealtimeFrameTensorMechanisms, RealtimeGenerationState,
    RealtimeHostTokenMaterializer, RealtimeIdentity, RealtimeMaterializationTask,
    RealtimeMechanism, RealtimeMechanismFacts, RealtimeMechanismSupport,
    RealtimeModelConstructionMechanisms, RealtimeModelSessionIdentity,
    RealtimeObservationRequirements, RealtimePayloadHistory, RealtimePayloadState,
    RealtimeSessionScheduler, SelectedRealtimeRealization, StateComponentPlacement,
    StaticParameterVisitorMut, SubmittedRealtimeFrame, WeightLoweringKind,
};
use std::{
    convert::Infallible,
    error::Error as _,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    rc::Rc,
    time::{Duration, Instant},
};

type State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Output = SubmittedRealtimeFrame<NumericTensor, NumericCompletion>;
type Sessions = RealtimeSessionScheduler<
    RealtimePayloadState<TransactionalState, NumericTensor>,
    StatefulNumericSampler,
    i32,
    NumericCompletion,
    Output,
>;
const REQUEST: RequestId = RequestId::new(91);
