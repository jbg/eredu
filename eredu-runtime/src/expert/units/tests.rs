use super::*;

#[derive(Default)]
struct Observer {
    starts: usize,
    finishes: Vec<bool>,
    reject_start: bool,
    reject_finish: bool,
}
impl RoutedUnitObserver<u64> for Observer {
    fn begin_invocation(&mut self, input: &RoutedUnitInvocation<'_, u64>) -> Result<(), Error> {
        assert_eq!(*input.input, 73);
        self.starts += 1;
        if self.reject_start {
            Err(Error::backend("start sentinel"))
        } else {
            Ok(())
        }
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        self.finishes.push(success);
        if self.reject_finish {
            Err(Error::backend("finish sentinel"))
        } else {
            Ok(())
        }
    }
    fn observe(&mut self, _: &RoutedUnitBatch<'_, u64>) -> Result<(), Error> {
        unreachable!()
    }
}
fn invocation() -> RoutedUnitInvocation<'static, u64> {
    RoutedUnitInvocation {
        input: &73,
        origins: None,
        unit_coordinates: None,
    }
}

#[test]
fn local_invocation_scope_suppresses_nested_callbacks_and_finishes_before_return() {
    let mut observer = Observer::default();
    let result = with_routed_unit_invocation(
        Some(&mut observer),
        invocation(),
        |nested| {
            assert!(nested.as_ref().unwrap().invocation_active());
            with_routed_unit_invocation(
                nested,
                invocation(),
                |nested| {
                    assert!(nested.unwrap().invocation_active());
                    Ok::<_, Error>(29)
                },
                |error| error,
            )
        },
        |error| error,
    )
    .unwrap();
    assert_eq!(result, 29);
    assert_eq!(observer.starts, 1);
    assert_eq!(observer.finishes, [true]);
}

#[test]
fn local_invocation_failure_still_finishes_and_preserves_the_first_cause() {
    for failure in ["start", "provider", "finish"] {
        let mut observer = Observer {
            reject_start: failure == "start",
            reject_finish: true,
            ..Default::default()
        };
        let mut calls = 0;
        let error = with_routed_unit_invocation(
            Some(&mut observer),
            invocation(),
            |_| {
                calls += 1;
                if failure == "provider" {
                    Err(Error::backend("provider sentinel"))
                } else {
                    Ok(())
                }
            },
            |error| error,
        )
        .unwrap_err();
        assert!(error.to_string().contains(&format!("{failure} sentinel")));
        assert_eq!(calls, usize::from(failure != "start"));
        assert_eq!(observer.starts, 1);
        assert_eq!(observer.finishes, [failure == "finish"]);
    }
}

#[test]
fn disabled_local_invocation_passes_through_without_an_observer() {
    assert_eq!(
        with_routed_unit_invocation(
            None,
            invocation(),
            |observer| {
                assert!(observer.is_none());
                Ok::<_, Error>(19)
            },
            |error| error
        )
        .unwrap(),
        19
    );
}
