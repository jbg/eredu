//! Module completion is required even when an equation changes state then fails.
use super::*;

#[derive(Default)]
struct Probe {
    roots: Vec<Vec<f32>>,
    fail: bool,
}
thread_local! {
    static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) };
}
struct Guard;
impl Guard {
    fn new(fail: bool) -> Self {
        PROBE.with(|probe| {
            *probe.borrow_mut() = Some(Probe {
                fail,
                ..Default::default()
            })
        });
        Self
    }
    fn roots(&self) -> Vec<Vec<f32>> {
        PROBE.with(|probe| probe.borrow().as_ref().unwrap().roots.clone())
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        PROBE.with(|probe| *probe.borrow_mut() = None);
    }
}

pub(super) fn complete<'a>(
    values: impl IntoIterator<Item = &'a NumericTensor>,
) -> Result<(), eredu_core::BackendFailure> {
    PROBE.with(|probe| {
        let mut probe = probe.borrow_mut();
        if let Some(probe) = probe.as_mut() {
            probe.roots.push(
                values
                    .into_iter()
                    .flat_map(|value| value.data.clone())
                    .collect(),
            );
            if probe.fail {
                return Err(eredu_core::BackendFailure::from_error(
                    std::io::Error::other("injected module completion failure"),
                ));
            }
        } else {
            for _ in values {}
        }
        Ok(())
    })
}

fn module(name: &str, context: &NumericContext) -> Module<NumericLinear> {
    let mut linear = NumericBackend::linear(
        LinearSpec {
            input: 2,
            output: 1,
            weight: ParameterSpec::trainable(name).unwrap(),
            bias: None,
            format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)
                .unwrap(),
        },
        context,
    )
    .unwrap();
    linear.weight = NumericTensor::new([1, 2], vec![2., -3.]);
    Module(linear)
}

#[test]
fn prediction_module_failure_settles_changed_state_before_returning() {
    let context = NumericContext::default();
    let mut module = module("prediction.weight", &context);
    let input = NumericTensor::new([1, 2], vec![4., 1.]);
    let mut state = None;
    let guard = Guard::new(false);
    let outcome = Materializer::invoke_module(&mut module, &context, |module| {
        state = Some(module.forward(&input, &context).unwrap());
        PredictionInvocation::new(
            Err::<(), _>(Error::backend("equation failed after cache update")),
            state.iter().cloned(),
        )
    });
    assert!(outcome
        .unwrap_err()
        .to_string()
        .contains("equation failed after cache update"));
    assert_eq!(guard.roots(), [vec![5.]]);
    assert_eq!(state.unwrap().data, [5.]);
}

#[test]
fn prediction_shared_owners_settle_the_same_result_and_state() {
    let context = NumericContext::default();
    let mut first = module("prediction.unit.weight", &context);
    let mut shared = module("prediction.shared.weight", &context);
    let guard = Guard::new(false);
    let outcome = Materializer::invoke_module_with_shared(
        &mut first,
        Some(&mut shared),
        &context,
        |unit, shared| {
            let input = NumericTensor::new([1, 2], vec![4., 1.]);
            let changed = unit.forward(&input, &context).unwrap();
            let output = shared
                .unwrap()
                .forward(
                    &NumericTensor::new([1, 2], vec![changed.data[0], 2.]),
                    &context,
                )
                .unwrap();
            PredictionInvocation::new(Ok(output.clone()), [changed, output])
        },
    )
    .unwrap();
    assert_eq!(outcome.data, [4.]);
    assert_eq!(guard.roots(), [vec![5., 4.], vec![5., 4.]]);
}

#[test]
fn prediction_module_completion_failure_prevents_success_publication() {
    use std::error::Error as _;
    let context = NumericContext::default();
    let mut module = module("prediction.weight", &context);
    let guard = Guard::new(true);
    let error = Materializer::invoke_module(&mut module, &context, |_| {
        PredictionInvocation::new(Ok(7), [NumericTensor::new([1], vec![9.])])
    })
    .unwrap_err();
    assert!(error
        .source()
        .unwrap()
        .to_string()
        .contains("injected module completion failure"));
    assert_eq!(guard.roots(), [vec![9.]]);
}
