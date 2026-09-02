//! Cold-path model-family selection and MLX session composition.

pub(crate) mod artifact;
pub mod automatic;
mod capability;
pub mod distributed;
mod execution;
mod inspection;
mod load_request;
pub mod loading;
mod model;
mod prepared_speculative;
#[cfg(any(feature = "image", feature = "audio"))]
mod processor;
mod realization;
pub mod realtime;
mod replicated_text;
mod session;
pub mod speculative;
pub mod structural;

pub use inspection::{inspect_model, MlxInspectionOptions};
pub use load_request::MlxLoadRequest;
pub(crate) use loading::validate_gguf_quantization_source;
pub use loading::{MlxModelConfig, MlxSelectedPreparation};
pub(crate) use model::Executable;
#[cfg(any(feature = "image", feature = "audio"))]
pub(crate) use processor::ModelProcessor;
pub use session::{MlxModelInput, MlxModelOutput, MlxModelSession, MlxSessionCompletion};

pub(crate) use crate::backend::{
    error::Error, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel,
};

#[cfg(test)]
pub(crate) mod path_instrumentation {
    use std::cell::Cell;

    #[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
    pub(crate) struct Counts {
        pub(crate) architecture_constructions: usize,
        pub(crate) state_allocations: usize,
        pub(crate) payload_opens: usize,
        pub(crate) forwards: usize,
    }

    thread_local! {
        static COUNTS: Cell<Counts> = Cell::new(Counts::default());
    }

    pub(crate) fn reset() {
        COUNTS.set(Counts::default());
    }

    pub(crate) fn snapshot() -> Counts {
        COUNTS.get()
    }

    pub(crate) fn architecture_construction() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.architecture_constructions += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn payload_open() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.payload_opens += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn state_allocation() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.state_allocations += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn forward() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.forwards += 1;
            cell.set(counts);
        });
    }
}
