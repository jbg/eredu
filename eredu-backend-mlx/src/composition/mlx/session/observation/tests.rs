use super::*;
use eredu_runtime::{
    with_routed_unit_invocation, RoutedUnitBatch, RoutedUnitInvocation, RoutedUnitObserver,
};

#[derive(Debug, thiserror::Error)]
#[error("composed native observer failure")]
struct Sentinel;

struct Observer<'a> {
    input: &'a MlxTensor,
    units: &'a eredu_core::component::ComponentCoordinateMap,
    active: bool,
    fail: bool,
    started: usize,
    finished: Vec<bool>,
}
impl RuntimeActivationObserver<MlxTensor, Error> for Observer<'_> {
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        unreachable!()
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<MlxTensor>>, Error> {
        assert_eq!(path, "layer.0.units");
        Ok(Some(self))
    }
}
impl RoutedUnitObserver<MlxTensor> for Observer<'_> {
    fn invocation_active(&self) -> bool {
        self.active
    }
    fn begin_invocation(
        &mut self,
        invocation: &RoutedUnitInvocation<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
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
        self.started += 1;
        if self.fail {
            Err(eredu_nn::Error::backend_source(Sentinel))
        } else {
            Ok(())
        }
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), eredu_nn::Error> {
        self.finished.push(success);
        if self.fail {
            Err(eredu_nn::Error::backend("later finish failure"))
        } else {
            Ok(())
        }
    }
    fn observe(&mut self, _: &RoutedUnitBatch<'_, MlxTensor>) -> Result<(), eredu_nn::Error> {
        unreachable!()
    }
}

#[test]
#[ignore = "requires native MLX access"]
fn composed_array_and_neutral_adapters_preserve_routed_invocation_scope() {
    let input = MlxTensor::from_array(Array::from_slice(&[0.3f32, -0.7], &[1, 2]));
    let units = eredu_core::component::ComponentCoordinateMap::indices(5, vec![3, 1]).unwrap();
    let origins = eredu_runtime::RoutedUnitOrigins::new(&[0, 1], &[3], 2).unwrap();
    for (active, fail) in [(false, false), (false, true), (true, false)] {
        let mut inner = Observer {
            input: &input,
            units: &units,
            active,
            fail,
            started: 0,
            finished: Vec::new(),
        };
        let mut arrays = ArrayObserverAdapter {
            inner: &mut inner,
            routed_path: None,
            routed_invocation_active: false,
        };
        let mut neutral = crate::composition::NeutralActivationObserver::new(&mut arrays);
        let observer = neutral.routed_unit_observer("layer.0.units").unwrap();
        assert_eq!(observer.as_ref().unwrap().invocation_active(), active);
        let mut called = false;
        let result = with_routed_unit_invocation(
            observer,
            RoutedUnitInvocation {
                input: &input,
                origins: Some(origins),
                unit_coordinates: Some(&units),
            },
            |observer| {
                assert!(observer.unwrap().invocation_active());
                called = true;
                Ok::<_, eredu_nn::Error>(())
            },
            |error| error,
        );
        assert_eq!(neutral.invocation_active(), active);
        assert_eq!(arrays.invocation_active(), active);
        assert_eq!(called, !fail);
        assert_eq!(inner.started, usize::from(!active));
        assert_eq!(inner.finished, if active { vec![] } else { vec![!fail] });
        if fail {
            let error = result.unwrap_err();
            let mut cause: &dyn std::error::Error = &error;
            while let Some(source) = cause.source() {
                cause = source;
            }
            assert!(cause.is::<Sentinel>());
        } else {
            result.unwrap();
        }
    }
}
