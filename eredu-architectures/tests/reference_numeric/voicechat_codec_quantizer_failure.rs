//! Local PRVQ preparation failures agree before tensor-parallel reductions.
use super::voicechat_codec_quantizer_residency::BoundFixturePolicy;
use super::voicechat_conditioning::compare;
use super::voicechat_mog_residency::weights;
use super::voicechat_rnnt_coordination::{coordinator_pair, Coordinator};
use super::voicechat_tts_generator::{codes, tensor};
use super::*;
use eredu_codec::{
    prvq::{DistanceMetric, Geometry},
    prvq_layered::{
        quantizer_parameter_description, LayerwiseQuantizer, QuantizerArchitecture,
        QuantizerForward, QuantizerInput,
    },
    Error as CodecError,
};
use eredu_runtime::{
    derive_partitioned_local_layout, replicated_execution::ReplicatedCoordinator,
    LayeredLocalPhase, LayeredTraversalHook,
};
use std::{cell::RefCell, rc::Rc, sync::atomic::Ordering};

struct Hook(Rc<RefCell<Coordinator>>);
impl LayeredTraversalHook<NumericBackend, QuantizerForward<NumericTensor>, CodecError> for Hook {
    fn local_phase(
        &mut self,
        phase: LayeredLocalPhase,
        success: bool,
        context: &NumericContext,
    ) -> Result<(), CodecError> {
        let (phase, group, index) = match phase {
            LayeredLocalPhase::Initialization => (0, 0, 0),
            LayeredLocalPhase::PolicyBegin => (1, 0, 0),
            LayeredLocalPhase::GroupIngress { group } => (2, group, 0),
            LayeredLocalPhase::BeforeUnit { group, index } => (3, group, index),
            LayeredLocalPhase::Acquisition { group, index } => (4, group, index),
            LayeredLocalPhase::UnitExecution { group, index } => (5, group, index),
            LayeredLocalPhase::UnitCompletion { group, index } => (6, group, index),
            LayeredLocalPhase::GroupCompletion { group } => (7, group, 0),
            LayeredLocalPhase::Finish => (8, 0, 0),
        };
        if !self.0.borrow_mut().agree(
            [
                0x50525651484f4f4b,
                phase,
                group as u64,
                index as u64,
                0,
                0,
                0,
                0,
            ],
            success,
            context,
        )? {
            return Err(CodecError::InvalidShape(
                "PRVQ peer failed local traversal phase".into(),
            ));
        }
        Ok(())
    }
}
#[test]
fn voicechat_codec_quantizer_tp_preparation_failures_agree_and_retry() {
    let (coordinators, timeouts) = coordinator_pair();
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, coordinator)| {
                let group = group.clone();
                scope.spawn(move || worker(rank, coordinator, group))
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
fn worker(rank: usize, coordinator: Coordinator, group: Arc<NumericParallelGroup>) {
    let fixture: serde_json::Value =
        serde_json::from_str(eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON).unwrap();
    let geometry = Geometry {
        depth: 3,
        cardinality: 5,
        channels: 4,
    };
    let prefix = "tts_model.audio_codec.prvq";
    let description = quantizer_parameter_description(geometry, prefix).unwrap();
    let layout = derive_partitioned_local_layout(
        &description,
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank).unwrap(),
    )
    .unwrap();
    let context = NumericContext::with_local_layout(layout.clone());
    let parallel = NumericParallelContext::new(rank, group);
    let parameters = weights(&fixture["parameters"])
        .into_iter()
        .map(|(name, value)| {
            let value =
                select_parameter(&value, layout.tensor(&name).unwrap().placement()).unwrap();
            (name, value)
        })
        .collect();
    let mut model = LayerwiseQuantizer::new(
        QuantizerArchitecture::new_parallel(geometry, prefix, &layout).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters,
            aborted: 0,
        },
    );
    let coordinator = Rc::new(RefCell::new(coordinator));
    let mut hook = Hook(coordinator.clone());
    let agree = |descriptor, success, context: &NumericContext| {
        coordinator
            .borrow_mut()
            .agree(descriptor, success, context)
            .map_err(CodecError::from)
    };
    let latent = tensor(&fixture["input"]);
    let recorded = codes(&fixture["codes"], &context);
    // Mismatched operation modes reject before acquisition or collective schedules diverge.
    let mismatch = if rank == 0 {
        QuantizerInput::Decode {
            codes: &recorded,
            range: 0..3,
        }
    } else {
        QuantizerInput::Quantize {
            latent: &latent,
            range: 0..3,
            metric: DistanceMetric::SquaredEuclidean,
        }
    };
    assert!(model
        .invoke_parallel_guarded(mismatch, &parallel, &context, &mut hook, agree)
        .is_err());
    assert!(model.runtime().policy().inner.events.is_empty());
    assert!(parallel.trace().is_empty());
    model.runtime_mut().policy_mut().inner.fail_acquire = (rank == 0).then_some(2);
    let before = parallel.trace().len();
    assert!(model
        .invoke_parallel_guarded(
            QuantizerInput::Quantize {
                latent: &latent,
                range: 0..3,
                metric: DistanceMetric::SquaredEuclidean
            },
            &parallel,
            &context,
            &mut hook,
            agree
        )
        .is_err());
    assert_eq!(
        parallel.trace().len() - before,
        2,
        "only completed codebook scores may reduce"
    );
    assert_eq!(model.runtime().policy().aborted, 1);
    model.runtime_mut().policy_mut().inner.fail_acquire = None;
    let (_, result) = model
        .invoke_parallel_guarded(
            QuantizerInput::Quantize {
                latent: &latent,
                range: 0..3,
                metric: DistanceMetric::SquaredEuclidean,
            },
            &parallel,
            &context,
            &mut hook,
            agree,
        )
        .unwrap();
    let result = result.into_quantized().unwrap();
    assert_eq!(result.codes.data, recorded.data);
    compare(&result.quantized, &fixture["quantized"], &fixture);
    compare(&result.residual, &fixture["residual"], &fixture);
    // The final depth can bind malformed local means after earlier local lookups.
    // No participant may enter the final channel reduction in that case.
    let key = format!("{prefix}.mus_list.2");
    let saved = model.runtime().policy().parameters[&key].clone();
    if rank == 0 {
        model
            .runtime_mut()
            .policy_mut()
            .parameters
            .insert(key.clone(), NumericTensor::zeros([5, 3]));
    }
    let before = parallel.trace().len();
    assert!(model
        .invoke_parallel_guarded(
            QuantizerInput::Decode {
                codes: &recorded,
                range: 0..3
            },
            &parallel,
            &context,
            &mut hook,
            agree
        )
        .is_err());
    assert_eq!(parallel.trace().len(), before);
    assert_eq!(model.runtime().policy().aborted, 2);
    model
        .runtime_mut()
        .policy_mut()
        .parameters
        .insert(key, saved);
    let (decoded, _) = model
        .invoke_parallel_guarded(
            QuantizerInput::Decode {
                codes: &recorded,
                range: 0..3,
            },
            &parallel,
            &context,
            &mut hook,
            agree,
        )
        .unwrap();
    compare(&decoded, &fixture["decoded"], &fixture);
    assert_eq!(latent.data, tensor(&fixture["input"]).data);
}
