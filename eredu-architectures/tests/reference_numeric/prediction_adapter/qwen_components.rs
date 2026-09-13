//! Prepared prediction captures and causal masks use the retained execution lane.
use super::*;
use eredu_runtime::ActivationObserver;

#[derive(Default)]
pub(super) struct Observe {
    pub values: BTreeMap<String, NumericTensor>,
    zero: Option<String>,
    zero_component: Option<usize>,
    fail: Option<String>,
    active_units: Option<eredu_core::component::ComponentCoordinateMap>,
    unit_starts: usize,
    unit_finishes: Vec<bool>,
    units: Vec<NumericTensor>,
    effective_units: Vec<NumericTensor>,
    zero_units: bool,
    fail_units: bool,
}
impl ActivationObserver<NumericTensor, Error> for Observe {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.values.insert(path.into(), value.clone());
        if self.fail.as_deref() == Some(path) {
            return Err(Error::backend(
                "injected prepared prediction observation failure",
            ));
        }
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        _: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<NumericTensor>>, Error> {
        Ok(Some(self))
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        if self.zero.as_deref() != Some(path) {
            return Ok(None);
        }
        let mut value = value.clone();
        let width = *value.shape.last().unwrap() as usize;
        let index = value.data.len() - width + self.zero_component.unwrap_or(1);
        value.data[index] = 0.0;
        Ok(Some(value))
    }
}

impl eredu_runtime::RoutedUnitObserver<NumericTensor> for Observe {
    fn begin_invocation(
        &mut self,
        invocation: &eredu_runtime::RoutedUnitInvocation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(self.active_units.is_none());
        assert!(invocation.origins.is_none());
        let coordinates = invocation
            .unit_coordinates
            .expect("prepared resident unit coordinates");
        assert_eq!(coordinates.contiguous_range().unwrap().start, 0);
        self.active_units = Some(coordinates.clone());
        self.unit_starts += 1;
        Ok(())
    }
    fn invocation_active(&self) -> bool {
        self.active_units.is_some()
    }
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(
            self.active_units.is_some(),
            "capture runs inside its provider scope"
        );
        assert!(
            batch.unit_coordinates.is_none(),
            "serial capture uses local coordinates"
        );
        assert!(batch.origins.is_none());
        assert!(batch.units.values.shape[1] > 0);
        self.units.push(batch.units.values.clone());
        if self.fail_units {
            return Err(Error::backend(
                "injected prepared prediction routed failure",
            ));
        }
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<Option<NumericTensor>, Error> {
        if !self.zero_units {
            return Ok(None);
        }
        let mut value = batch.units.values.clone();
        for row in value
            .data
            .chunks_exact_mut(batch.units.values.shape[1] as usize)
        {
            row[self.zero_component.unwrap_or(1)] = 0.0;
        }
        Ok(Some(value))
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(self.active_units.is_some());
        self.effective_units.push(batch.units.values.clone());
        Ok(())
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        assert!(self.active_units.take().is_some());
        self.unit_finishes.push(success);
        Ok(())
    }
}

struct Parameters(BTreeMap<String, NumericTensor>);
impl<'a> eredu_nn::ParameterVisitor<'a, NumericTensor> for Parameters {
    fn visit(&mut self, metadata: eredu_nn::ParameterMetadata, value: &'a NumericTensor) {
        assert!(self
            .0
            .insert(metadata.id.to_string(), value.clone())
            .is_none());
    }
}
impl PredictionModuleVisitor<NumericBackend, Materializer> for Parameters {
    type Error = std::convert::Infallible;
    fn visit<M: Parameterized<NumericTensor>>(
        &mut self,
        _: usize,
        module: &mut Module<M>,
    ) -> Result<(), Self::Error> {
        module.as_mut().visit_parameters(self);
        Ok(())
    }
}

fn project(input: &NumericTensor, weight: &NumericTensor) -> NumericTensor {
    let width = weight.shape[1] as usize;
    NumericTensor::new(
        [1, input.shape[1], weight.shape[0]],
        input
            .data
            .chunks_exact(width)
            .flat_map(|row| {
                weight.data.chunks_exact(width).map(move |weights| {
                    row.iter()
                        .zip(weights)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>() as f32
                })
            })
            .collect(),
    )
}

pub(super) fn verify<A, E, D>(
    scopes: &[eredu_core::component::ComponentExecutionScope],
    extension: &mut E,
    invoker: &mut Invoker<'_, A, D>,
    checkpoint: &E::LaneState,
    prior: &NumericTensor,
    token: &NumericTensor,
    expected: &NumericTensor,
    expected_capture: &NumericTensor,
) -> Result<(), String>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Clone,
    D: Driver<A>,
    E: MaterializedPredictionExecutor<A, NumericBackend, Materializer>,
{
    use eredu_core::component::ComponentResidualBase;
    let mut parameters = Parameters(BTreeMap::new());
    extension.visit_modules(&mut parameters).unwrap();
    assert_eq!(scopes.len(), extension.depth());
    for (depth, scope) in scopes.iter().enumerate() {
        let mut plain = checkpoint.clone();
        let (ordinary, ordinary_capture) =
            extension.logits::<State, _>(invoker, prior, token, depth, &mut plain)?;
        if depth == 0 {
            assert_tensor_exact(&ordinary, expected, "prepared ordinary replay logits");
            assert_tensor_exact(
                &ordinary_capture,
                expected_capture,
                "prepared ordinary replay hidden",
            );
        }
        let mut record = Observe::default();
        let mut lane = checkpoint.clone();
        let (actual, capture) = extension.logits_observed::<State, _>(
            invoker,
            prior,
            token,
            depth,
            &mut lane,
            Some(&mut record),
        )?;
        assert_tensor_exact(&actual, &ordinary, "prepared prediction no-op logits");
        assert_tensor_exact(
            &capture,
            &ordinary_capture,
            "prepared prediction no-op hidden",
        );
        assert_tensor_exact(
            &record.values[&scope.readout.logits],
            &ordinary,
            "actual readout output",
        );
        assert_tensor_exact(
            &record.values[&format!("{}.effective", scope.readout.normalized)],
            &ordinary_capture,
            "actual retained prediction hidden",
        );
        for component in &scope.components {
            let input = &record.values[&component.effective_activation];
            let mut write = project(input, &parameters.0[&component.write_weight]);
            if let Some(bias) = &component.write_bias {
                write = write.add(&parameters.0[bias], invoker.context).unwrap();
            }
            assert_tensor_close(
                &write,
                &record.values[component.write_output.as_ref().unwrap()],
                "prepared component sum reconstructs actual write",
            );
        }
        let ComponentResidualBase::LinearFusion {
            inputs,
            weight,
            projection_input,
            output,
            ..
        } = &scope.residual_base
        else {
            panic!("linear prediction fusion")
        };
        let parts = inputs
            .iter()
            .map(|input| record.values[&input.output].clone())
            .collect::<Vec<_>>();
        let joined = NumericTensor::concatenate(&parts, -1, invoker.context).unwrap();
        assert_tensor_exact(
            &joined,
            &record.values[projection_input],
            "declared fusion input order",
        );
        assert_tensor_close(
            &project(&joined, &parameters.0[weight]),
            &record.values[output],
            "declared fusion reconstructs actual residual base",
        );

        if !scope.routed_components.is_empty() {
            assert_eq!(record.unit_starts, scope.routed_components.len());
            assert_eq!(
                record.unit_finishes,
                vec![true; scope.routed_components.len()]
            );
            assert!(record.active_units.is_none());
            assert!(!record.units.is_empty());
            for (before, after) in record.units.iter().zip(&record.effective_units) {
                assert_tensor_exact(before, after, "routed no-op preserves units");
            }
            let active = record
                .units
                .iter()
                .find_map(|value| {
                    value
                        .data
                        .iter()
                        .enumerate()
                        .find(|(_, value)| **value != 0.0)
                        .map(|(index, _)| index % value.shape[1] as usize)
                })
                .expect("nonzero routed fixture");
            let mut masked = checkpoint.clone();
            let mut trace = Observe {
                zero_units: true,
                zero_component: Some(active),
                ..Default::default()
            };
            let (edited, _) = extension.logits_observed::<State, _>(
                invoker,
                prior,
                token,
                depth,
                &mut masked,
                Some(&mut trace),
            )?;
            assert_eq!(
                trace.unit_finishes,
                vec![true; scope.routed_components.len()]
            );
            assert_eq!(trace.units.len(), trace.effective_units.len());
            for (before, after) in trace.units.iter().zip(&trace.effective_units) {
                for (index, (&before_value, &after_value)) in
                    before.data.iter().zip(&after.data).enumerate()
                {
                    assert_eq!(
                        after_value,
                        if index % before.shape[1] as usize == active {
                            0.0
                        } else {
                            before_value
                        }
                    );
                }
            }
            assert_ne!(
                edited.data, ordinary.data,
                "routed units causally affect prediction"
            );
            let mut failed = checkpoint.clone();
            let mut trace = Observe {
                fail_units: true,
                ..Default::default()
            };
            assert!(extension
                .logits_observed::<State, _>(
                    invoker,
                    prior,
                    token,
                    depth,
                    &mut failed,
                    Some(&mut trace)
                )
                .unwrap_err()
                .contains("injected prepared prediction routed failure"));
            assert_eq!(trace.unit_finishes, [false]);
            assert!(
                trace.active_units.is_none(),
                "failed provider finishes its scope"
            );
        }
        let paths = inputs
            .iter()
            .map(|input| input.output.strip_suffix(".effective").unwrap().to_owned())
            .chain([output.clone()])
            .chain(
                scope
                    .components
                    .iter()
                    .map(|component| component.activation.clone()),
            )
            .chain([
                scope.readout.normalized.clone(),
                scope.readout.linear_scores.clone(),
            ]);
        for path in paths {
            let baseline = &record.values[&path];
            assert_eq!(baseline.shape[1], 1);
            let selected = baseline
                .data
                .iter()
                .position(|value| *value != 0.0)
                .expect("nonzero mask fixture");
            let mut trace = Observe {
                zero: Some(path.clone()),
                zero_component: Some(selected),
                ..Default::default()
            };
            let mut branch = checkpoint.clone();
            let (edited, _) = extension.logits_observed::<State, _>(
                invoker,
                prior,
                token,
                depth,
                &mut branch,
                Some(&mut trace),
            )?;
            let before = &trace.values[&path];
            let after = &trace.values[&format!("{path}.effective")];
            for (index, (&a, &b)) in before.data.iter().zip(&after.data).enumerate() {
                assert_eq!(
                    b,
                    if index == selected { 0.0 } else { a },
                    "only selected coordinate changes: {path}"
                );
            }
            assert_ne!(edited.data, ordinary.data, "causal prepared mask: {path}");
        }
        let mut failed = checkpoint.clone();
        let mut trace = Observe {
            fail: Some(scope.readout.linear_scores.clone()),
            ..Default::default()
        };
        assert!(extension
            .logits_observed::<State, _>(
                invoker,
                prior,
                token,
                depth,
                &mut failed,
                Some(&mut trace)
            )
            .unwrap_err()
            .contains("injected prepared prediction"));
        let mut replay = checkpoint.clone();
        let (replayed, replay_capture) =
            extension.logits::<State, _>(invoker, prior, token, depth, &mut replay)?;
        assert_tensor_exact(
            &replayed,
            &ordinary,
            "failed observation and siblings leave checkpoint unchanged",
        );
        assert_tensor_exact(
            &replay_capture,
            &ordinary_capture,
            "prediction hidden replay",
        );
    }
    let mut advanced = checkpoint.clone();
    let mut trace = Observe::default();
    extension.advance_observed::<State, _>(
        invoker,
        prior,
        token,
        &mut advanced,
        Some(&mut trace),
    )?;
    assert_tensor_exact(
        &trace.values[&scopes[0].readout.logits],
        expected,
        "accepted-token replay uses the observed prediction driver",
    );
    Ok(())
}
