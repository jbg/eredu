use super::*;
use crate::{
    capture::{CaptureBackendProvider, SpeculativeCaptureObserver},
    inspection::{with_speculative_activation, SpeculativeActivationObserver},
    ActivationObserver,
};
use eredu_core::speculative::*;
use eredu_nn::routing_intervention::{GroupSelectionAction, GroupSelectionControl};

// Actual integer IDs and F32 coefficients preserve the admitted Preview source
// categories; a floating-point stand-in is not an integer source proof.
#[derive(Clone)]
struct RouteValue {
    shape: Vec<u64>,
    data: RouteData,
}
#[derive(Clone)]
enum RouteData {
    I32(Vec<i32>),
    F32(Vec<f32>),
}
impl RouteValue {
    fn ids(&self) -> &[i32] {
        let RouteData::I32(values) = &self.data else {
            panic!("IDs");
        };
        values
    }
    fn weights(&self) -> &[f32] {
        let RouteData::F32(values) = &self.data else {
            panic!("coefficients");
        };
        values
    }
}
#[derive(Default)]
struct RoutingBackend {
    copies: usize,
}
impl CaptureBackend for RoutingBackend {
    type Tensor = RouteValue;
    type Error = std::io::Error;
    fn source_dtype(&self, value: &RouteValue) -> Option<eredu_core::checkpoint::TensorDtype> {
        Some(match &value.data {
            RouteData::I32(_) => eredu_core::checkpoint::TensorDtype::I32,
            RouteData::F32(_) => eredu_core::checkpoint::TensorDtype::F32,
        })
    }
    fn shape(&self, value: &RouteValue) -> Result<Vec<u64>, Self::Error> {
        Ok(value.shape.clone())
    }
    fn estimate(
        &self,
        value: &RouteValue,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate(&value.shape, selection, slice)
    }
    fn transform(
        &mut self,
        value: &RouteValue,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.copies += 1;
        let CaptureTransform::Preview { max_elements } = selection.transform else {
            panic!("routing Preview only");
        };
        let selected: Vec<_> = indices(&value.shape, slice)
            .into_iter()
            .take(max_elements as usize)
            .collect();
        let data = match &value.data {
            RouteData::I32(values) => {
                TensorObservationData::I64(selected.iter().map(|i| i64::from(values[*i])).collect())
            }
            RouteData::F32(values) => {
                TensorObservationData::F32(selected.iter().map(|i| values[*i]).collect())
            }
        };
        Ok(CapturePayload::Tensor(
            TensorObservation::new(vec![selected.len()], data).unwrap(),
        ))
    }
}
impl InterventionBackend for RoutingBackend {
    fn intervention_dtype(&self, _: &RouteValue) -> Result<InterventionDtype, std::io::Error> {
        panic!("no dense callback")
    }
    fn validate_intervention_geometry(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        panic!("no dense callback")
    }
    fn select_region(
        &mut self,
        _: &RouteValue,
        _: &ResolvedCaptureSlice,
    ) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn update_region(
        &mut self,
        _: &RouteValue,
        _: &ResolvedCaptureSlice,
        _: &RouteValue,
    ) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn zeros(&mut self, _: &[u64], _: InterventionDtype) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn scale(&mut self, _: &RouteValue, _: f32) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn fill_masked(
        &mut self,
        _: &RouteValue,
        _: &[bool],
        _: f32,
    ) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn mask_components(
        &mut self,
        _: &RouteValue,
        _: &[u32],
        _: bool,
    ) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn realize_tensor(&mut self, _: &InterventionTensor) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn add(&mut self, _: &RouteValue, _: &RouteValue) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
    fn fill_columns(
        &mut self,
        _: &RouteValue,
        _: &[u32],
        _: f32,
    ) -> Result<RouteValue, std::io::Error> {
        panic!("no dense callback")
    }
}
struct Provider;
impl CaptureBackendProvider for Provider {
    type Tensor = RouteValue;
    type Error = std::io::Error;
    type Backend<'a> = RoutingBackend;
    fn backend(&mut self) -> RoutingBackend {
        RoutingBackend::default()
    }
}
fn transport(e: &CaptureExecutionError<std::io::Error>) -> String {
    e.to_string()
}
type Observer =
    SpeculativeCaptureObserver<Provider, fn(&CaptureExecutionError<std::io::Error>) -> String>;
fn run(action: InterventionAction, stride: u64) -> CaptureSession {
    let (capture, old) = routed(false);
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 5,
        max_context: None,
        max_predictions: 1,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        completeness: DescriptionCompleteness::Complete,
        points: vec![],
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: CaptureCapabilities::default(),
        points: vec![],
    };
    let capture = capture
        .plan()
        .clone()
        .admit_invocations(&catalog, &support, &support.capture, bounds)
        .unwrap();
    let mut discovery = discovery(&old);
    discovery.points[0].operations.push(action.kind());
    let mut plan = old.plan().clone();
    plan.operations[0].action = action;
    plan.operations[0].slices = vec![CaptureSlice {
        axis: "token".into(),
        start: 1,
        end: 4,
        stride,
    }];
    // Existing routing admission requires bounded ID/coefficient previews.
    // The actual I32/F32 source fields retain the same bounded ordered prefix.
    plan.operations[0].evidence = InterventionEvidence::Preview { max_elements: 3 };
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(
            plan.admit_invocations(&discovery, bounds, "session")
                .unwrap(),
            Arc::new(Facts::new(0)),
        )
        .unwrap();
    session
}
fn decision(a: u64, b: u64) -> (RouteValue, RouteValue) {
    (
        RouteValue {
            shape: vec![b - a, 2],
            data: RouteData::I32(
                (a..b)
                    .flat_map(|r| [(r % 4) as i32, ((r + 1) % 4) as i32])
                    .collect(),
            ),
        },
        RouteValue {
            shape: vec![b - a, 2],
            data: RouteData::F32(
                (a..b)
                    .flat_map(|r| [0.2 + r as f32 * 0.05, 0.8 - r as f32 * 0.05])
                    .collect(),
            ),
        },
    )
}
fn changed(
    control: &GroupSelectionControl,
    ids: &RouteValue,
    weights: &RouteValue,
) -> (RouteValue, RouteValue) {
    let (mut ids, mut weights) = (ids.clone(), weights.clone());
    let RouteData::I32(id_values) = &mut ids.data else {
        panic!("IDs");
    };
    let RouteData::F32(weight_values) = &mut weights.data else {
        panic!("coefficients");
    };
    for (ordinal, row) in (control.first_row..control.end_row)
        .step_by(control.row_stride as usize)
        .enumerate()
    {
        for k in 0..2 {
            let i = row as usize * 2 + k;
            match &control.action {
                GroupSelectionAction::Force(values) => {
                    id_values[i] = values[ordinal * 2 + k] as i32
                }
                GroupSelectionAction::ZeroContribution(values)
                    if values.contains(&(id_values[i] as u32)) =>
                {
                    weight_values[i] = 0.
                }
                _ => (),
            }
        }
    }
    (ids, weights)
}
fn drive(
    o: &mut dyn ActivationObserver<RouteValue, String>,
    a: u64,
    b: u64,
) -> Result<(Vec<i32>, Vec<f32>), String> {
    let control = o.routing_control("router", b - a)?;
    let (ids, weights) = decision(a, b);
    if let Some(control) = control {
        let (new_ids, new_weights) = changed(&control, &ids, &weights);
        o.routing_applied(
            "router",
            Some(crate::RoutingDecision {
                ids: &ids,
                coefficients: &weights,
            }),
            crate::RoutingDecision {
                ids: &new_ids,
                coefficients: &new_weights,
            },
        )?;
        Ok((new_ids.ids().to_vec(), new_weights.weights().to_vec()))
    } else {
        o.routing_unmodified(
            "unrelated",
            crate::RoutingDecision {
                ids: &ids,
                coefficients: &weights,
            },
        )?;
        o.routing_unmodified(
            "router",
            crate::RoutingDecision {
                ids: &ids,
                coefficients: &weights,
            },
        )?;
        Ok((ids.ids().to_vec(), weights.weights().to_vec()))
    }
}
#[test]
fn routing_windows_preserve_force_payload_ordinals_no_overlap_and_four_evidence_fields() {
    for action in [
        InterventionAction::ForceExperts {
            shape: [2, 2],
            expert_ids: vec![3, 1, 2, 0],
        },
        InterventionAction::ZeroExpertContribution {
            expert_ids: vec![1],
        },
    ] {
        let mut ordinary = run(action.clone(), 2);
        ordinary
            .begin_invocation(
                CapturePhase::Prefill,
                0,
                CaptureInvocationShape {
                    batch: 1,
                    sequence: 5,
                    context: None,
                },
                crate::capture::CaptureInvocationSelection::default(),
            )
            .unwrap();
        let expected = {
            let mut o = CaptureObserver::new(
                &mut ordinary,
                RoutingBackend::default(),
                |e: CaptureExecutionError<std::io::Error>| e.to_string(),
            );
            let v = drive(&mut o, 0, 5).unwrap();
            o.finish().unwrap();
            v
        };
        let expected_step = ordinary.take_step().unwrap();
        let mut observer = SpeculativeCaptureObserver::new(
            run(action, 2),
            Provider,
            transport as fn(&CaptureExecutionError<std::io::Error>) -> String,
            SpeculativeRequestId::new(0),
            vec![],
            vec![SpeculativeCaptureScope::Target],
        )
        .unwrap();
        observer.set_activation_origin(Some(SpeculativeActivationOrigin {
            request: SpeculativeRequestId::new(0),
            committed_tokens: 0,
            prediction: 0,
            prefix_digest: [8; 32],
            optimistic: false,
        }));
        observer.set_prefill_reduction_geometry(SpeculativePrefillReductionGeometry {
            target_sequence: 5,
            prediction_sequence: 5,
        });
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        for (a, b) in [(0, 1), (1, 3), (3, 4), (4, 5)] {
            observer.set_prefill_span(Some(SpeculativePrefillSpan {
                prompt_tokens: 5,
                input_start: a,
                input_end: b,
                position: a,
                hidden_start: a,
                token_start: a,
                sequence: b - a,
                seed_start: a,
            }));
            let (i, w) = with_speculative_activation(
                Some(&mut observer),
                SpeculativeActivationPhase::TargetPrefill,
                (b - a) as usize,
                |o| drive(o.unwrap(), a, b),
            )
            .unwrap();
            ids.extend(i);
            weights.extend(w);
        }
        observer.complete_prefill_reductions().unwrap();
        observer.finish_prefill_reductions(true);
        assert_eq!((ids, weights), expected);
        let steps: Vec<_> = std::iter::from_fn(|| observer.take_activation_capture()).collect();
        assert_eq!(
            steps[0].captures.as_step().interventions[0].outcome,
            InterventionOutcome::Unmatched
        );
        assert_eq!(
            steps[3].captures.as_step().interventions[0].outcome,
            InterventionOutcome::Unmatched
        );
        let total = &steps[3].prefill_reductions.as_ref().unwrap().interventions[0];
        assert_eq!(total.status, SpeculativePrefillReductionStatus::Complete);
        assert_eq!(total.covered_sequence, 5);
        assert_eq!(total.record.evidence.len(), 4);
        for (a, b) in total
            .record
            .evidence
            .iter()
            .zip(&expected_step.interventions[0].evidence)
        {
            assert_eq!(a.source_shape, b.source_shape);
            assert_eq!(a.selected_shape, b.selected_shape);
            assert_eq!(a.source_dtype, b.source_dtype);
            assert_eq!(a.outcome, b.outcome);
            assert_eq!(a.payload, b.payload);
            let Some(CapturePayload::Tensor(value)) = &a.payload else {
                panic!("preview");
            };
            assert_eq!(value.shape(), &[3]);
        }
    }
}
#[test]
fn no_overlap_notifications_cannot_resolve_missing_or_changed_control() {
    for mode in 0..5 {
        let mut session = run(
            InterventionAction::ZeroExpertContribution {
                expert_ids: vec![1],
            },
            2,
        );
        session
            .begin_invocation_window(
                CapturePhase::Prefill,
                0,
                CaptureInvocationShape {
                    batch: 1,
                    sequence: 1,
                    context: None,
                },
                crate::capture::CaptureInvocationSelection::default(),
                CaptureInvocationWindow {
                    logical_sequence: 5,
                    start: 0,
                },
            )
            .unwrap();
        assert!(session.routing_control("router", 1).unwrap().is_none());
        assert_eq!(
            session.routing_unmodified_interest("router"),
            crate::RoutingUnmodifiedInterest::Metadata
        );
        assert_eq!(
            session.routing_unmodified_interest("other"),
            crate::RoutingUnmodifiedInterest::None
        );
        let (ids, weights) = decision(0, 1);
        let value = || crate::RoutingDecision {
            ids: &ids,
            coefficients: &weights,
        };
        let mut backend = RoutingBackend::default();
        match mode {
            0 => {
                let usage = session.cumulative_usage();
                session
                    .routing_unmodified(&mut backend, "other", value())
                    .unwrap();
                assert_eq!(session.cumulative_usage(), usage);
                assert!(session.finish_interventions().is_err());
                session
                    .routing_unmodified(&mut backend, "router", value())
                    .unwrap();
                session.finish_interventions().unwrap();
                assert_eq!(backend.copies, 0);
            }
            1 => {
                assert!(session
                    .routing_applied(&mut backend, "router", Some(value()), value())
                    .is_err());
                assert!(session.finish_interventions().is_err());
            }
            2 => {
                session.invocation_window = Some(CaptureInvocationWindow {
                    logical_sequence: 5,
                    start: 1,
                });
                assert!(session
                    .routing_unmodified(&mut backend, "router", value())
                    .is_err());
            }
            3 => {
                session.phase = CapturePhase::Decode;
                assert!(session
                    .routing_unmodified(&mut backend, "router", value())
                    .is_err());
            }
            _ => {
                session.routing_failed("router", "selector failed");
                assert!(session.finish_interventions().is_err());
                assert!(matches!(
                    session.take_step().unwrap().interventions[0].outcome,
                    InterventionOutcome::Failed { .. }
                ));
            }
        }
    }
}
