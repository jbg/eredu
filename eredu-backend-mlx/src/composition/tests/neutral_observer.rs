use super::NeutralActivationObserver;
use crate::{backend::error::Error as MlxError, MlxTensor};
use eredu_core::component::ComponentCoordinateMap;
use eredu_runtime::{
    with_routed_unit_invocation, ActivationObserver, RoutedUnitBatch, RoutedUnitInvocation,
    RoutedUnitObserver, RoutedUnitOrigins,
};
use safemlx::Array;

#[derive(Debug, thiserror::Error)]
#[error("native observer sentinel: {0}")]
struct Sentinel(&'static str);

struct Observer<'a> {
    input: &'a Array,
    units: &'a ComponentCoordinateMap,
    inherited_active: bool,
    failure: &'static str,
    starts: usize,
    finishes: Vec<bool>,
}
impl ActivationObserver<Array, MlxError> for Observer<'_> {
    fn observe(&mut self, _: &str, _: &Array) -> Result<(), MlxError> {
        unreachable!()
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<Array>>, MlxError> {
        assert_eq!(path, "model.layers.0.mlp.experts.units");
        Ok(Some(self))
    }
}
impl RoutedUnitObserver<Array> for Observer<'_> {
    fn invocation_active(&self) -> bool {
        self.inherited_active
    }
    fn begin_invocation(
        &mut self,
        invocation: &RoutedUnitInvocation<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
        // The adapter must lend the same arrays/maps, not clone or materialize.
        assert!(std::ptr::eq(invocation.input, self.input));
        assert!(std::ptr::eq(
            invocation.unit_coordinates.unwrap(),
            self.units
        ));
        let origin = invocation.origins.unwrap().resolve(0).unwrap();
        assert_eq!(
            (origin.source_peer, origin.token, origin.slot),
            (Some(1), 1, 1)
        );
        self.starts += 1;
        if self.failure == "start" {
            Err(eredu_nn::Error::backend_retained_source(Sentinel("start")))
        } else {
            Ok(())
        }
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), eredu_nn::Error> {
        self.finishes.push(success);
        if self.failure != "none" {
            Err(eredu_nn::Error::backend_retained_source(Sentinel("finish")))
        } else {
            Ok(())
        }
    }
    fn observe(&mut self, _: &RoutedUnitBatch<'_, Array>) -> Result<(), eredu_nn::Error> {
        unreachable!()
    }
}

#[test]
#[ignore = "requires native MLX access"]
fn native_adapter_forwards_borrowed_invocation_and_preserves_first_failure() {
    let input = MlxTensor::from_array(Array::from_slice(&[0.3f32, -0.7], &[1, 2]));
    let units = ComponentCoordinateMap::indices(5, vec![3, 1]).unwrap();
    let origins = RoutedUnitOrigins::new(&[0, 1], &[3], 2).unwrap();
    for failure in ["none", "start", "provider", "finish"] {
        let mut inner = Observer {
            input: input.as_array(),
            units: &units,
            inherited_active: false,
            failure,
            starts: 0,
            finishes: Vec::new(),
        };
        let mut adapter = NeutralActivationObserver::new(&mut inner);
        let observer = adapter
            .routed_unit_observer("model.layers.0.mlp.experts.units")
            .unwrap();
        assert!(!observer.as_ref().unwrap().invocation_active());
        let mut calls = 0;
        let result = with_routed_unit_invocation(
            observer,
            RoutedUnitInvocation {
                input: &input,
                origins: Some(origins),
                unit_coordinates: Some(&units),
            },
            |observer| {
                calls += 1;
                assert!(observer.unwrap().invocation_active());
                if failure == "provider" {
                    Err(eredu_nn::Error::backend_retained_source(Sentinel("provider")))
                } else {
                    Ok(())
                }
            },
            |error| error,
        );
        assert!(!adapter.invocation_active());
        assert_eq!(inner.starts, 1);
        assert_eq!(calls, usize::from(failure != "start"));
        assert_eq!(inner.finishes, [matches!(failure, "none" | "finish")]);
        if failure == "none" {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            let mut cause: &dyn std::error::Error = &error;
            while let Some(source) = cause.source() {
                cause = source;
            }
            assert_eq!(cause.downcast_ref::<Sentinel>().unwrap().0, failure);
        }
    }
}

#[test]
#[ignore = "requires native MLX access"]
fn native_adapter_preserves_an_already_active_inner_scope() {
    let input = MlxTensor::from_array(Array::from_slice(&[0.3f32, -0.7], &[1, 2]));
    let units = ComponentCoordinateMap::indices(5, vec![3, 1]).unwrap();
    let mut inner = Observer {
        input: input.as_array(),
        units: &units,
        inherited_active: true,
        failure: "none",
        starts: 0,
        finishes: Vec::new(),
    };
    let mut adapter = NeutralActivationObserver::new(&mut inner);
    let observer = adapter
        .routed_unit_observer("model.layers.0.mlp.experts.units")
        .unwrap();
    with_routed_unit_invocation(
        observer,
        RoutedUnitInvocation {
            input: &input,
            origins: None,
            unit_coordinates: None,
        },
        |observer| {
            assert!(observer.unwrap().invocation_active());
            Ok::<_, eredu_nn::Error>(())
        },
        |error| error,
    )
    .unwrap();
    assert_eq!(inner.starts, 0);
    assert!(inner.finishes.is_empty());
}
