//! Actual prepared partition session capture transactions, without a second forward driver.
use super::*;
use eredu_runtime::{ForwardCaptureBundle, ReplicatedTextExecutionStrategy, ReplicatedTextSession};
type State = DeviceState<NumericBackend, NumericHybridLayerState>;

struct Tap {
    paths: BTreeSet<String>,
    values: std::rc::Rc<RefCell<BTreeMap<String, NumericTensor>>>,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Tap {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if self.paths.contains(path)
            && self
                .values
                .borrow_mut()
                .insert(path.into(), value.clone())
                .is_some()
        {
            return Err(Error::backend("capture output invoked twice"));
        }
        Ok(())
    }
    fn observe_replica(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.observe(path, value)
    }
}
pub(super) fn run<A, D>(
    session: &mut ReplicatedTextSession<A, NumericBackend, NumericReplicatedMechanisms, D>,
    input: A::Input<'_>,
    prefill: bool,
    bundle: &ForwardCaptureBundle,
    fault: Option<&'static str>,
    context: &NumericContext,
) -> Result<(NumericTensor, Vec<NumericTensor>), Error>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    D: ReplicatedTextExecutionStrategy<
        A,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    let values = std::rc::Rc::new(RefCell::new(BTreeMap::new()));
    let mut observer = Tap {
        paths: bundle
            .slots()
            .iter()
            .map(|slot| slot.role().to_owned())
            .collect(),
        values: values.clone(),
    };
    let rank = context.partition.as_ref().unwrap().rank;
    let collect = |_: &A::ForwardContext| {
        if fault == Some("collect") && rank == 1 {
            return Err(Error::backend("injected capture collection"));
        }
        bundle
            .slots()
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let value = if rank == slot.owner() {
                    values.borrow().get(slot.role()).cloned().ok_or_else(|| {
                        Error::backend(format!(
                            "missing selected capture {} on rank {rank}",
                            slot.role()
                        ))
                    })?
                } else {
                    NumericTensor::zeros(slot.shape().to_vec())
                };
                Ok(if fault == Some("geometry") && rank == 1 && index == 0 {
                    NumericTensor::zeros(vec![1, 1, 1])
                } else {
                    value
                })
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let assemble = |_: &A::ForwardContext, values| {
        if fault == Some("assemble") && rank == 1 {
            Err(Error::backend("injected capture assembly"))
        } else {
            Ok(values)
        }
    };
    let input = if fault == Some("input") && rank == 1 {
        Err(Error::backend("injected capture input preparation"))
    } else {
        Ok(input)
    };
    // A failed admission has no geometry from which to resolve a capture bundle.
    let selected_bundle = if input.is_err() { None } else { Some(bundle) };
    let result = if prefill {
        session.prefill_input_with_capture(
            input,
            context,
            &mut observer,
            selected_bundle,
            collect,
            assemble,
        )
    } else {
        session.decode_input_with_capture(
            input,
            context,
            &mut observer,
            selected_bundle,
            collect,
            assemble,
        )
    };
    if fault == Some("input") {
        assert!(
            values.borrow().is_empty(),
            "peer admission failure precedes architecture traversal"
        );
    }
    result.map_err(Error::backend_source)
}

/// Borrows the existing canonical executor while a speculative lane owns its KV.
pub(super) fn run_lane<A, D>(
    session: &mut ReplicatedTextSession<A, NumericBackend, NumericReplicatedMechanisms, D>,
    input: A::Input<'_>,
    prefill: bool,
    bundle: &ForwardCaptureBundle,
    state: &mut State,
    context: &NumericContext,
) -> Result<(NumericTensor, Vec<NumericTensor>), Error>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    D: ReplicatedTextExecutionStrategy<
        A,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    session
        .exchange_prediction_target_state(state, context)
        .map_err(Error::backend_source)?;
    match run(session, input, prefill, bundle, None, context) {
        Ok(value) => {
            session
                .exchange_prediction_target_state(state, context)
                .map_err(Error::backend_source)?;
            Ok(value)
        }
        Err(error) => {
            session
                .recover_prediction_target_state_after_failure(state)
                .map_err(Error::backend_source)?;
            Err(error)
        }
    }
}
