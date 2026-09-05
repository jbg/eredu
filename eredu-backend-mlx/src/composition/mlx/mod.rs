//! Cold-path neutral execution selection and MLX mechanism composition.

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
mod processor;
pub mod realtime;
pub(crate) mod replicated_text;
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
        pub(crate) constructors: usize,
        pub(crate) unit_constructions: usize,
        pub(crate) materializations: usize,
        pub(crate) local_static_bindings: usize,
        pub(crate) excluded_local_static_parameters: usize,
        pub(crate) forwards: usize,
        pub(crate) state_publications: usize,
        pub(crate) completions: usize,
    }

    thread_local! {
        static COUNTS: Cell<Counts> = Cell::new(Counts::default());
        static COMMUNICATION_REALIZATION_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
        static MANIFEST_COMMUNICATION_REALIZATION_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
        static NEUTRAL_PARTITIONED_CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
        static BOUNDED_UNIT_ACQUISITIONS: Cell<usize> = const { Cell::new(0) };
        static VARIABLE_ALL_TO_ALL_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
        static TARGET_NATIVE_RESOURCE_REALIZATION_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
    }

    pub(crate) fn reset() {
        COUNTS.set(Counts::default());
        COMMUNICATION_REALIZATION_ATTEMPTS.set(0);
        MANIFEST_COMMUNICATION_REALIZATION_ATTEMPTS.set(0);
        NEUTRAL_PARTITIONED_CONSTRUCTIONS.set(0);
        BOUNDED_UNIT_ACQUISITIONS.set(0);
        VARIABLE_ALL_TO_ALL_SUBMISSIONS.set(0);
        TARGET_NATIVE_RESOURCE_REALIZATION_ATTEMPTS.set(0);
    }

    pub(crate) fn snapshot() -> Counts {
        COUNTS.get()
    }

    pub(crate) fn communication_realization_attempts() -> usize {
        COMMUNICATION_REALIZATION_ATTEMPTS.get()
    }

    pub(crate) fn communication_realization_attempt() {
        COMMUNICATION_REALIZATION_ATTEMPTS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn target_native_resource_realization_attempts() -> usize {
        TARGET_NATIVE_RESOURCE_REALIZATION_ATTEMPTS.get()
    }

    pub(crate) fn target_native_resource_realization_attempt() {
        TARGET_NATIVE_RESOURCE_REALIZATION_ATTEMPTS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn manifest_communication_realization_attempts() -> usize {
        MANIFEST_COMMUNICATION_REALIZATION_ATTEMPTS.get()
    }

    pub(crate) fn manifest_communication_realization_attempt() {
        MANIFEST_COMMUNICATION_REALIZATION_ATTEMPTS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn architecture_construction() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.architecture_constructions += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn neutral_partitioned_construction() {
        NEUTRAL_PARTITIONED_CONSTRUCTIONS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn neutral_partitioned_constructions() -> usize {
        NEUTRAL_PARTITIONED_CONSTRUCTIONS.get()
    }

    pub(crate) fn bounded_unit_acquisitions() -> usize {
        BOUNDED_UNIT_ACQUISITIONS.get()
    }

    pub(crate) fn bounded_unit_acquisition() {
        BOUNDED_UNIT_ACQUISITIONS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn variable_all_to_all_submissions() -> usize {
        VARIABLE_ALL_TO_ALL_SUBMISSIONS.get()
    }

    pub(crate) fn variable_all_to_all_submission() {
        VARIABLE_ALL_TO_ALL_SUBMISSIONS.with(|count| count.set(count.get() + 1));
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

    pub(crate) fn constructor() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.constructors += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn unit_construction() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.unit_constructions += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn materialization() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.materializations += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn local_static_materialization(selected: usize, excluded: usize) {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.local_static_bindings += selected;
            counts.excluded_local_static_parameters += excluded;
            cell.set(counts);
        });
    }

    pub(crate) fn state_publication() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.state_publications += 1;
            cell.set(counts);
        });
    }

    pub(crate) fn completion() {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            counts.completions += 1;
            cell.set(counts);
        });
    }
}
