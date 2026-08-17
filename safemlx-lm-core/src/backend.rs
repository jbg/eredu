//! High-level contract implemented once per execution backend.

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{
    checkpoint::TensorDtype,
    topology::{ParallelAxis, ParallelTopology},
};

/// Stable, extensible description of an execution backend.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendDescriptor {
    /// Backend implementation name, such as `mlx` or `iree`.
    pub name: String,
    /// Backend implementation version.
    pub version: String,
}

/// Portable description of one backend-visible device.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceDescriptor {
    /// Backend-stable device identifier.
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Backend-specific device family without a closed core enum.
    pub family: String,
    /// Total memory when discoverable.
    pub memory_bytes: Option<u64>,
}

/// Fail-closed capabilities discovered from a backend and device.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Supports exact completion observation for submissions.
    pub exact_completion: bool,
    /// Supports device-to-device transfer for backend-owned values.
    pub transfers: bool,
    /// Supports collective execution for a complete session.
    pub collectives: bool,
    /// Supports backend-managed persistent decode caches.
    pub persistent_cache: bool,
}

/// Fail-closed distributed operations exposed by one selected session.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DistributedCapabilities {
    /// World-scoped collectives are available.
    pub world_collectives: bool,
    /// Active topology axes with subgroup collective support.
    pub collective_axes: Vec<ParallelAxis>,
    /// Point-to-point transfers are available.
    pub point_to_point: bool,
    /// Variable-count all-to-all exchange is available.
    pub variable_all_to_all: bool,
    /// Collective and transfer submissions have exact completion objects.
    pub exact_completion: bool,
}

/// Scope of a collective or point-to-point operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "axis", rename_all = "snake_case")]
pub enum CollectiveScope {
    /// All ranks in the selected distributed session.
    World,
    /// The topology subgroup containing this rank on one axis.
    Axis(ParallelAxis),
}

/// Portable shape and element type for a backend-owned received value.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValueDescriptor {
    /// Row-major logical shape. An empty shape describes a scalar.
    pub shape: Vec<usize>,
    /// Logical element type.
    pub dtype: TensorDtype,
}

/// Portable identity of one selected distributed session.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct DistributedSessionDescriptor {
    /// Backend-neutral Cartesian topology.
    pub topology: ParallelTopology,
    /// World rank represented by this process-local session.
    pub rank: usize,
}

impl DistributedSessionDescriptor {
    /// Validates that `rank` belongs to `topology`.
    pub fn new(topology: ParallelTopology, rank: usize) -> Result<Self, BackendError> {
        let topology = ParallelTopology::new(
            topology.tensor,
            topology.pipeline,
            topology.expert,
            topology.data,
        )
        .map_err(|error| BackendError::Preparation {
            operation: "distributed session topology".into(),
            message: error.to_string(),
        })?;
        if rank >= topology.world_size() {
            return Err(BackendError::Preparation {
                operation: "distributed session topology".into(),
                message: format!(
                    "rank {rank} is outside topology world size {}",
                    topology.world_size()
                ),
            });
        }
        Ok(Self { topology, rank })
    }
}

impl<'de> Deserialize<'de> for DistributedSessionDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawDescriptor {
            topology: ParallelTopology,
            rank: usize,
        }

        let raw = RawDescriptor::deserialize(deserializer)?;
        Self::new(raw.topology, raw.rank).map_err(serde::de::Error::custom)
    }
}

/// Structured backend failure that does not expose a runtime exception type.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum BackendError {
    /// A required capability is absent.
    #[error("backend {backend} does not support required capability {capability}")]
    Unsupported {
        /// Backend implementation name.
        backend: String,
        /// Required capability.
        capability: String,
    },
    /// Model preparation failed.
    #[error("backend model preparation failed during {operation}: {message}")]
    Preparation {
        /// Preparation operation.
        operation: String,
        /// Backend-provided context.
        message: String,
    },
    /// Session execution failed.
    #[error("backend session {session} failed during {operation}: {message}")]
    Execution {
        /// Stable session identifier.
        session: String,
        /// Operation being executed.
        operation: String,
        /// Backend-provided context.
        message: String,
    },
    /// Exact completion observation failed.
    #[error("backend completion observation failed: {message}")]
    Completion {
        /// Backend-provided context.
        message: String,
    },
}

/// Exact completion owned by one backend submission.
pub trait Completion {
    /// Error produced while observing the completion.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Nonblocking exact-completion observation.
    fn is_complete(&self) -> Result<bool, Self::Error>;

    /// Blocks on this exact completion only.
    fn wait(&self) -> Result<(), Self::Error>;
}

/// Output and exact completion returned by a backend submission.
#[derive(Debug)]
pub struct Submission<T, C> {
    /// Backend-owned output value.
    pub output: T,
    /// Completion retaining everything needed by the submitted work.
    pub completion: C,
}

impl<T, C> Submission<T, C>
where
    C: Completion,
{
    /// Waits for this exact submission and returns its output.
    pub fn wait(self) -> Result<T, C::Error> {
        self.completion.wait()?;
        Ok(self.output)
    }
}

/// Marker wrapper proving that a model was prepared by a backend.
#[derive(Debug)]
pub struct PreparedModel<M> {
    model: M,
}

impl<M> PreparedModel<M> {
    /// Wraps a backend-prepared model.
    pub const fn new(model: M) -> Self {
        Self { model }
    }
    /// Borrows the backend model.
    pub const fn get(&self) -> &M {
        &self.model
    }
    /// Mutably borrows the backend model.
    pub fn get_mut(&mut self) -> &mut M {
        &mut self.model
    }
    /// Consumes the marker.
    pub fn into_inner(self) -> M {
        self.model
    }
}

/// One backend selected for an entire prepared model and all its sessions.
pub trait Backend: Sized {
    /// Portable model preparation request.
    type ModelConfig;
    /// Opaque backend model/executable.
    type Model;
    /// Opaque backend session/cache state and execution implementation.
    type Session: BackendSession<Self>;
    /// Backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Backend identity.
    fn descriptor(&self) -> BackendDescriptor;
    /// Discovers devices and their fail-closed capabilities.
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, BackendCapabilities)>, Self::Error>;
    /// Loads, compiles, or materializes a model for this backend.
    fn prepare_model(
        &self,
        config: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error>;
    /// Creates backend-owned session/cache state for a prepared model.
    fn create_session(
        &self,
        model: &PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error>;
}

/// Prefill/decode interface for an already selected backend session.
///
/// The contract intentionally models language-model submissions rather than
/// primitive tensor operations. Input, output, cache and completion stay opaque.
pub trait BackendSession<B: Backend> {
    /// Backend-owned prefill input.
    type PrefillInput;
    /// Backend-owned decode input.
    type DecodeInput;
    /// Backend-owned logits/output.
    type Output;
    /// Exact completion type.
    type Completion: Completion<Error = B::Error>;

    /// Submits prompt prefill against this session.
    fn prefill(
        &mut self,
        backend: &B,
        model: &mut B::Model,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error>;

    /// Submits one or more cached decode positions against this session.
    fn decode(
        &mut self,
        backend: &B,
        model: &mut B::Model,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error>;
}

/// Optional high-level transfer and collective capability of a selected session.
///
/// This contract deliberately operates on an opaque backend value. It models
/// the few communication submissions needed by model execution without making
/// core define a tensor algebra or exposing native groups, streams, or events.
/// Every operation is scoped to the session selected for the complete model.
pub trait DistributedSession {
    /// Backend-owned tensor or buffer value.
    type Value;
    /// Exact completion retaining the submitted communication resources.
    type Completion: Completion<Error = Self::Error>;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable topology and rank identity.
    fn descriptor(&self) -> DistributedSessionDescriptor;
    /// Fail-closed communication support.
    fn capabilities(&self) -> DistributedCapabilities;

    /// Submits a sum reduction over `scope`.
    fn all_reduce_sum(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a leading-rank-axis gather over `scope`.
    fn all_gather(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a variable-count all-to-all exchange over `scope`.
    fn all_to_all_v(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
        send_counts: &[usize],
        receive_counts: &[usize],
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a point-to-point send to a rank within `scope`.
    fn send(
        &self,
        scope: CollectiveScope,
        peer: usize,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a point-to-point receive from a rank within `scope`.
    fn receive(
        &self,
        scope: CollectiveScope,
        peer: usize,
        value: &ValueDescriptor,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Synchronously gathers portable scheduler metadata across the world.
    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error>;
}

/// Backend extension exposing communication attached to a model session.
pub trait DistributedBackend: Backend {
    /// Selected distributed session implementation.
    type DistributedSession: DistributedSession<Error = Self::Error>;

    /// Returns communication for a distributed model session.
    fn distributed_session<'a>(session: &'a Self::Session) -> Option<&'a Self::DistributedSession>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[derive(Debug, Clone, Copy)]
    struct Done;
    impl Completion for Done {
        type Error = Infallible;
        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }
        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    struct Mock;
    impl Backend for Mock {
        type ModelConfig = u32;
        type Model = u32;
        type Session = MockSession;
        type Error = Infallible;
        fn descriptor(&self) -> BackendDescriptor {
            BackendDescriptor {
                name: "mock".into(),
                version: "1".into(),
            }
        }
        fn devices(&self) -> Result<Vec<(DeviceDescriptor, BackendCapabilities)>, Self::Error> {
            Ok(vec![])
        }
        fn prepare_model(&self, config: u32) -> Result<PreparedModel<u32>, Self::Error> {
            Ok(PreparedModel::new(config))
        }
        fn create_session(&self, _: &PreparedModel<u32>) -> Result<MockSession, Self::Error> {
            Ok(MockSession {
                tokens: vec![],
                distributed: None,
            })
        }
    }
    struct MockSession {
        tokens: Vec<u32>,
        distributed: Option<MockDistributed>,
    }
    impl BackendSession<Mock> for MockSession {
        type PrefillInput = Vec<u32>;
        type DecodeInput = u32;
        type Output = u32;
        type Completion = Done;
        fn prefill(
            &mut self,
            _: &Mock,
            model: &mut u32,
            input: Vec<u32>,
        ) -> Result<Submission<u32, Done>, Infallible> {
            self.tokens.extend(input);
            Ok(Submission {
                output: self.tokens.len() as u32 + *model,
                completion: Done,
            })
        }
        fn decode(
            &mut self,
            _: &Mock,
            model: &mut u32,
            input: u32,
        ) -> Result<Submission<u32, Done>, Infallible> {
            self.tokens.push(input);
            Ok(Submission {
                output: self.tokens.len() as u32 + *model,
                completion: Done,
            })
        }
    }

    #[test]
    fn mock_prefill_and_multiple_decode_steps() {
        let backend = Mock;
        let prepared = backend.prepare_model(10).unwrap();
        let mut session = backend.create_session(&prepared).unwrap();
        let mut model = prepared.into_inner();
        let prefill = session.prefill(&backend, &mut model, vec![1, 2]).unwrap();
        assert_eq!(prefill.output, 12);
        assert!(prefill.completion.is_complete().unwrap());
        assert_eq!(session.decode(&backend, &mut model, 3).unwrap().output, 13);
        assert_eq!(session.decode(&backend, &mut model, 4).unwrap().output, 14);
    }

    #[derive(Debug, Clone, Copy)]
    struct MockDistributed {
        descriptor: DistributedSessionDescriptor,
    }

    impl DistributedSession for MockDistributed {
        type Value = Vec<u32>;
        type Completion = Done;
        type Error = Infallible;

        fn descriptor(&self) -> DistributedSessionDescriptor {
            self.descriptor.clone()
        }

        fn capabilities(&self) -> DistributedCapabilities {
            DistributedCapabilities {
                world_collectives: true,
                collective_axes: vec![ParallelAxis::Tensor],
                point_to_point: true,
                variable_all_to_all: true,
                exact_completion: true,
            }
        }

        fn all_reduce_sum(
            &self,
            _: CollectiveScope,
            input: &Vec<u32>,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: input.iter().map(|value| value * 2).collect(),
                completion: Done,
            })
        }

        fn all_gather(
            &self,
            _: CollectiveScope,
            input: &Vec<u32>,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            let mut output = input.clone();
            output.extend(input);
            Ok(Submission {
                output,
                completion: Done,
            })
        }

        fn all_to_all_v(
            &self,
            _: CollectiveScope,
            input: &Vec<u32>,
            _: &[usize],
            _: &[usize],
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: input.clone(),
                completion: Done,
            })
        }

        fn send(
            &self,
            _: CollectiveScope,
            _: usize,
            input: &Vec<u32>,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: input.clone(),
                completion: Done,
            })
        }

        fn receive(
            &self,
            _: CollectiveScope,
            peer: usize,
            value: &ValueDescriptor,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: vec![peer as u32; value.shape.iter().product()],
                completion: Done,
            })
        }

        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Infallible> {
            let mut output = local.to_vec();
            output.extend_from_slice(local);
            Ok(output)
        }
    }

    impl DistributedBackend for Mock {
        type DistributedSession = MockDistributed;

        fn distributed_session<'a>(
            session: &'a MockSession,
        ) -> Option<&'a Self::DistributedSession> {
            session.distributed.as_ref()
        }
    }

    #[test]
    fn mock_distributed_session_owns_collective_and_transfer_lifecycle() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let session = MockDistributed {
            descriptor: DistributedSessionDescriptor::new(topology, 0).unwrap(),
        };
        let capabilities = session.capabilities();
        assert!(capabilities.exact_completion);
        assert_eq!(capabilities.collective_axes, vec![ParallelAxis::Tensor]);
        assert_eq!(
            session
                .all_reduce_sum(CollectiveScope::Axis(ParallelAxis::Tensor), &vec![2, 3])
                .unwrap()
                .wait()
                .unwrap(),
            vec![4, 6]
        );
        assert_eq!(
            session
                .receive(
                    CollectiveScope::World,
                    1,
                    &ValueDescriptor {
                        shape: vec![2],
                        dtype: TensorDtype::U32,
                    },
                )
                .unwrap()
                .wait()
                .unwrap(),
            vec![1, 1]
        );
        assert_eq!(session.all_gather_words(&[7]).unwrap(), vec![7, 7]);

        let model_session = MockSession {
            tokens: Vec::new(),
            distributed: Some(session),
        };
        assert_eq!(
            Mock::distributed_session(&model_session)
                .unwrap()
                .descriptor(),
            session.descriptor()
        );
    }

    #[test]
    fn distributed_descriptors_round_trip_and_reject_invalid_ranks() {
        let descriptor =
            DistributedSessionDescriptor::new(ParallelTopology::new(2, 3, 1, 1).unwrap(), 4)
                .unwrap();
        let encoded = serde_json::to_string(&descriptor).unwrap();
        assert_eq!(
            serde_json::from_str::<DistributedSessionDescriptor>(&encoded).unwrap(),
            descriptor
        );
        let scope = CollectiveScope::Axis(ParallelAxis::Pipeline);
        assert_eq!(
            serde_json::from_str::<CollectiveScope>(&serde_json::to_string(&scope).unwrap())
                .unwrap(),
            scope
        );
        assert!(DistributedSessionDescriptor::new(descriptor.topology, 6).is_err());
        assert!(serde_json::from_str::<DistributedSessionDescriptor>(
            r#"{"topology":{"tensor":2,"pipeline":3,"expert":1,"data":1},"rank":6}"#
        )
        .is_err());
    }
}
