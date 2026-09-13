//! Internal observation admission at the actual typed invocation boundary.
use eredu_core::speculative::SpeculativeActivationPhase;
use eredu_runtime::inspection::{
    with_speculative_activation, ObserverErrorBridge, SpeculativeActivationObserver,
};

pub(super) fn neural<T, E: std::fmt::Display, R>(
    observer: Option<&mut dyn SpeculativeActivationObserver<T, E>>,
    phase: SpeculativeActivationPhase,
    sequence: usize,
    to_observer: impl FnMut(eredu_nn::Error) -> E,
    operation: impl FnOnce(
        Option<&mut dyn eredu_runtime::ActivationObserver<T, eredu_nn::Error>>,
    ) -> Result<R, E>,
) -> Result<R, E> {
    with_speculative_activation(observer, phase, sequence, |observer| {
        let Some(observer) = observer else {
            return operation(None);
        };
        let mut bridge = ObserverErrorBridge::new(observer, to_observer, |error: &E| {
            eredu_nn::Error::backend(error.to_string())
        });
        let result = operation(Some(&mut bridge));
        bridge.resolve(result)
    })
}
