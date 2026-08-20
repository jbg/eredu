//! Backend-neutral activation and routed-expert observation contracts.

/// Normalized routed-expert data emitted by architecture implementations.
pub struct RoutingObservation<'a, T> {
    /// Stable path-like name of the routed block.
    pub path: &'a str,
    /// Selected expert IDs shaped `[..., top_k]`.
    pub selected_experts: &'a T,
    /// Selected scores before optional top-k renormalization.
    pub selected_scores: &'a T,
    /// Final route weights applied to expert outputs.
    pub route_weights: &'a T,
    /// Combined routed expert contribution.
    pub routed_output: &'a T,
    /// Rank-local contribution before expert-parallel reduction.
    pub local_routed_output: Option<&'a T>,
    /// Globally reduced expert-parallel contribution.
    pub reduced_routed_output: Option<&'a T>,
    /// Shared-expert contribution when the architecture has one.
    pub shared_output: Option<&'a T>,
    /// Combined routed and shared contribution when reported separately.
    pub combined_output: Option<&'a T>,
    /// Total number of routed experts.
    pub expert_count: i32,
}

/// Statically dispatched activation observation and intervention contract.
pub trait ActivationObserver<T, E> {
    /// Observes a named backend-native tensor.
    fn observe(&mut self, path: &str, value: &T) -> Result<(), E>;

    /// Optionally replaces an activation before downstream computation.
    fn intervene(&mut self, _path: &str, _value: &T) -> Result<Option<T>, E> {
        Ok(None)
    }

    /// Observes normalized routed-expert decisions and contributions.
    fn observe_routing(&mut self, _routing: RoutingObservation<'_, T>) -> Result<(), E> {
        Ok(())
    }
}

/// Observes an activation and applies an optional replacement without
/// materializing its backend-native value.
pub fn observe_and_intervene<T, E, O>(observer: &mut O, path: &str, value: &T) -> Result<T, E>
where
    T: Clone,
    O: ActivationObserver<T, E> + ?Sized,
{
    observer.observe(path, value)?;
    Ok(observer
        .intervene(path, value)?
        .unwrap_or_else(|| value.clone()))
}

/// Zero-sized observer used by the ordinary unobserved inference path.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopObserver;

impl<T, E> ActivationObserver<T, E> for NoopObserver {
    fn observe(&mut self, _path: &str, _value: &T) -> Result<(), E> {
        Ok(())
    }
}

impl<T, E, F> ActivationObserver<T, E> for F
where
    F: FnMut(&str, &T) -> Result<(), E>,
{
    fn observe(&mut self, path: &str, value: &T) -> Result<(), E> {
        self(path, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_observer_is_static_and_passthrough() {
        fn observe<O: ActivationObserver<i32, ()>>(observer: &mut O) {
            observer.observe("layer.output", &7).unwrap();
            assert_eq!(observer.intervene("layer.output", &7).unwrap(), None);
        }
        observe(&mut NoopObserver);
    }

    #[test]
    fn observed_activation_can_be_replaced_without_an_erased_hot_path() {
        struct ReplacingObserver {
            observed: Vec<String>,
        }

        impl ActivationObserver<i32, ()> for ReplacingObserver {
            fn observe(&mut self, path: &str, _value: &i32) -> Result<(), ()> {
                self.observed.push(path.into());
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &i32) -> Result<Option<i32>, ()> {
                Ok((path == "model.layers.0.output").then_some(value + 4))
            }
        }

        let mut observer = ReplacingObserver {
            observed: Vec::new(),
        };
        let output = observe_and_intervene(&mut observer, "model.layers.0.output", &3).unwrap();
        assert_eq!(output, 7);
        assert_eq!(observer.observed, ["model.layers.0.output"]);
    }
}
