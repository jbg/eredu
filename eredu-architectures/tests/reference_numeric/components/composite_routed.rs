//! Shared routed component evidence for composite family conformance.
use super::*;

pub(super) struct RoutedComponentObserver<'a> {
    pub(super) dense: GlobalComponentObserver<'a>,
    pub(super) path: String,
    pub(super) sparse_mask: bool,
    pub(super) original: Option<NumericTensor>,
    pub(super) rows: usize,
    pub(super) parameters: Option<&'a BTreeMap<String, NumericTensor>>,
    pub(super) masked_path: &'static str,
    pub(super) route_suffix: &'static str,
    pub(super) weight_suffix: &'static str,
    pub(super) write_suffix: &'static str,
    pub(super) projected: BTreeMap<String, Vec<f64>>,
}
impl ActivationObserver<NumericTensor, Error> for RoutedComponentObserver<'_> {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if let Some(sum) = self.projected.remove(path) {
            assert_tensor_close(
                &NumericTensor::new(
                    value.shape.clone(),
                    sum.into_iter().map(|v| v as f32).collect(),
                ),
                value,
                "selected sparse components reconstruct the complete write before postnorm",
            );
        }
        self.dense.observe(path, value)
    }
    fn observe_replica(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.dense.observe_replica(path, value)
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        self.dense.intervene(path, value)
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<NumericTensor>>, Error> {
        self.path = path.into();
        if let Some(layout) = self.dense.layout {
            assert!(layout
                .routed_observation(&format!("{path}.units"))
                .unwrap()
                .ownership()
                .is_some());
        }
        Ok(Some(self))
    }
}
impl RoutedComponentObserver<'_> {
    fn edit(
        &self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Option<NumericTensor> {
        if !self.sparse_mask || self.path != self.masked_path {
            return None;
        }
        let mut value = batch.units.values.clone();
        let width = value.shape[1] as usize;
        let selected_width = *batch.units.coefficients.shape.last().unwrap() as usize;
        let coordinates = batch.unit_coordinates;
        for row in 0..value.shape[0] as usize {
            let token = batch.units.token_indices.data[row] as usize;
            let slot = batch.units.selection_indices.data[row] as usize % selected_width;
            if batch.route_origin(token, slot).unwrap().token != self.dense.position {
                continue;
            }
            for column in 0..width {
                if coordinates.map_or(Some(column), |map| map.local_to_global(column)) != Some(1) {
                    value.data[row * width + column] = 0.0;
                }
            }
        }
        Some(value)
    }
}
impl eredu_runtime::RoutedUnitObserver<NumericTensor> for RoutedComponentObserver<'_> {
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        self.rows += batch.units.values.shape[0] as usize;
        assert!(self.original.replace(batch.units.values.clone()).is_none());
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<Option<NumericTensor>, Error> {
        Ok(self.edit(batch))
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        let original = self.original.take().unwrap();
        let before = eredu_runtime::RoutedUnitBatch {
            units: batch.units.with_values(&original),
            ..*batch
        };
        let expected = self.edit(&before).unwrap_or(original);
        assert_tensor_exact(
            batch.units.values,
            &expected,
            "sparse effective values follow global token/unit coordinates",
        );
        if let Some(parameters) = self.parameters {
            let prefix = self.path.strip_suffix(self.route_suffix).unwrap();
            let weight = &parameters[&format!("{prefix}{}", self.weight_suffix)];
            let hidden = weight.shape[1] as usize;
            let width = weight.shape[2] as usize;
            assert_eq!(batch.units.values.shape[1] as usize, width);
            let sum = self
                .projected
                .entry(format!("{prefix}{}", self.write_suffix))
                .or_insert_with(|| vec![0.0; batch.units.total_token_count * hidden]);
            let routes = *batch.units.coefficients.shape.last().unwrap() as usize;
            let source_routes = *batch.source_groups.shape.last().unwrap() as usize;
            for (row, units) in batch.units.values.data.chunks_exact(width).enumerate() {
                let native_token = batch.units.token_indices.data[row] as usize;
                let selected = batch.units.selection_indices.data[row] as usize;
                let slot = selected % routes;
                let source_token = batch.source_token(native_token).unwrap();
                let expert = batch
                    .global_group(
                        batch.source_groups.data[source_token * source_routes + slot] as usize,
                    )
                    .unwrap();
                let token = batch.route_origin(native_token, slot).unwrap().token;
                let coefficient = f64::from(batch.units.coefficients.data[selected]);
                for channel in 0..hidden {
                    sum[token * hidden + channel] += coefficient
                        * units
                            .iter()
                            .zip(
                                &weight.data[(expert * hidden + channel) * width
                                    ..(expert * hidden + channel + 1) * width],
                            )
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>();
                }
            }
        }
        Ok(())
    }
}
