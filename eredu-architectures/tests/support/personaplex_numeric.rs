//! Private reduced PersonaPlex equations; never a released-profile parser option.
#![allow(dead_code, unused_imports)]
use crate as eredu_architectures;
include!("numeric/imports.rs");
include!("numeric/dense_format.rs");
include!("numeric/backend.rs");
include!("numeric/sampler.rs");
include!("numeric/assertions.rs");
include!("numeric/payload_read.rs");
include!("numeric/parameter_backend.rs");
#[path = "../reference_numeric/rotary.rs"]
mod rotary;
mod grouped_linear {
    use super::*;
    use eredu_nn::{GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec};
    include!("numeric/grouped_linear.rs");
}
mod routed_units {
    use super::*;
    use eredu_nn::{GroupedUnitBatch, GroupedUnitObserver};
    include!("numeric/routed_units.rs");
}
mod payload {
    use super::*;
    use eredu_checkpoint::{recipe::RecipeDtype, store::CheckpointSource, StoredDtype};
    include!("numeric/payload.rs");
}
thread_local! {
    static PAYLOAD_READS: RefCell<Vec<ReferencePayloadRead>> = const { RefCell::new(Vec::new()) };
}
fn record_reference_payload_reads(reads: Vec<ReferencePayloadRead>) {
    PAYLOAD_READS.with(|events| events.borrow_mut().extend(reads));
}
mod frames {
    include!("numeric/frame_imports.rs");
    include!("numeric/frame_support.rs");
    include!("numeric/frame_selection.rs");
    include!("numeric/frames.rs");
    include!("numeric/tensor_file.rs");
    include!("numeric/vocabulary.rs");
    mod residency {
        include!("numeric/frame_residency.rs");
    }
    mod low_support {
        use super::*;
        use eredu_core::OutputDemand;
        include!("numeric/frame_low.rs");
        pub(super) fn run_low(
            path: &std::path::Path,
            config: &moshi::MoshiConfig,
            residency: ExecutionResidency,
            demand: OutputDemand,
            diagnostics: bool,
            observed: bool,
        ) -> Vec<LowStep> {
            let source = moshi::prepare_selected_moshi_realtime_source(selected_with_residency(
                path, config, residency,
            ))
            .unwrap();
            source.artifact_identity().unwrap();
            moshi::visit_selected_moshi_realtime_architecture::<NumericBackend, State, _>(
                source,
                &NumericContext::default(),
                LowVisitor {
                    demand,
                    diagnostics,
                    observed,
                    config: config.clone(),
                    widths: vec![2, 3, 1, 1, 1],
                    token_card: [
                        config.text_vocabulary_size(),
                        config.audio_vocabulary_size(),
                    ],
                },
            )
            .unwrap()
        }
        include!("numeric/personaplex_low_test.rs");
    }
    include!("numeric/personaplex_frames.rs");
}
