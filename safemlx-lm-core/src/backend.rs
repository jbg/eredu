//! High-level contract implemented once per execution backend.

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
pub trait Backend {
    /// Portable model preparation request.
    type ModelConfig;
    /// Opaque backend model/executable.
    type Model;
    /// Opaque backend session/cache state.
    type Session;
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
        backend: &B,
        model: &mut PreparedModel<B::Model>,
        session: &mut B::Session,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error>;

    /// Submits one or more cached decode positions against this session.
    fn decode(
        backend: &B,
        model: &mut PreparedModel<B::Model>,
        session: &mut B::Session,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error>;
}

/// Borrowing executor used by an established model/session owner.
///
/// Facades can use this contract without moving a public model into an opaque
/// container. It has the same whole-session selection rule as [`Backend`].
pub trait SessionExecutor {
    /// Backend-owned prefill input.
    type PrefillInput;
    /// Backend-owned decode input.
    type DecodeInput;
    /// Backend-owned output.
    type Output;
    /// Exact completion type.
    type Completion: Completion<Error = Self::Error>;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Submits prompt prefill.
    fn prefill(
        &mut self,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error>;

    /// Submits cached decode.
    fn decode(
        &mut self,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error>;
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
        type Session = Vec<u32>;
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
        fn create_session(&self, _: &PreparedModel<u32>) -> Result<Vec<u32>, Self::Error> {
            Ok(vec![])
        }
    }
    struct MockSession;
    impl BackendSession<Mock> for MockSession {
        type PrefillInput = Vec<u32>;
        type DecodeInput = u32;
        type Output = u32;
        type Completion = Done;
        fn prefill(
            _: &Mock,
            model: &mut PreparedModel<u32>,
            session: &mut Vec<u32>,
            input: Vec<u32>,
        ) -> Result<Submission<u32, Done>, Infallible> {
            session.extend(input);
            Ok(Submission {
                output: session.len() as u32 + *model.get(),
                completion: Done,
            })
        }
        fn decode(
            _: &Mock,
            model: &mut PreparedModel<u32>,
            session: &mut Vec<u32>,
            input: u32,
        ) -> Result<Submission<u32, Done>, Infallible> {
            session.push(input);
            Ok(Submission {
                output: session.len() as u32 + *model.get(),
                completion: Done,
            })
        }
    }

    #[test]
    fn mock_prefill_and_multiple_decode_steps() {
        let backend = Mock;
        let mut model = backend.prepare_model(10).unwrap();
        let mut session = backend.create_session(&model).unwrap();
        let prefill = MockSession::prefill(&backend, &mut model, &mut session, vec![1, 2]).unwrap();
        assert_eq!(prefill.output, 12);
        assert!(prefill.completion.is_complete().unwrap());
        assert_eq!(
            MockSession::decode(&backend, &mut model, &mut session, 3)
                .unwrap()
                .output,
            13
        );
        assert_eq!(
            MockSession::decode(&backend, &mut model, &mut session, 4)
                .unwrap()
                .output,
            14
        );
    }
}
