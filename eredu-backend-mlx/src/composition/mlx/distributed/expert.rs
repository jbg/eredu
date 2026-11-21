//! MLX routed-expert mechanisms shared by distributed composition.

use eredu_nn::tensor_movement::ExpertRouteTensorMovement;
use safemlx::{ops::zeros_dtype, Array, Stream};

use crate::{backend::error::Error, MlxTensor};

/// MLX arbitrary-row movement for architecture-owned expert exchange.
#[derive(Debug, Clone)]
pub struct MlxExpertRouteTensorMovement {
    stream: Stream,
}

impl MlxExpertRouteTensorMovement {
    /// Binds tensor movement to one caller-owned execution stream.
    pub(crate) fn new(stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn indices(&self, values: &[usize], trailing_axis: bool) -> Result<Array, Error> {
        let values = values
            .iter()
            .copied()
            .map(|value| {
                i32::try_from(value).map_err(|_| {
                    Error::ArchitectureModel("expert route index exceeds MLX i32 indexing".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rows = i32::try_from(values.len()).map_err(|_| {
            Error::ArchitectureModel("expert route index count exceeds MLX i32 geometry".into())
        })?;
        let shape = if trailing_axis {
            vec![rows, 1]
        } else {
            vec![rows]
        };
        Array::from_slice(&values, &shape)
            .copy(&self.stream)
            .map_err(Into::into)
    }
}

impl ExpertRouteTensorMovement<MlxTensor> for MlxExpertRouteTensorMovement {
    type Error = Error;

    fn shape(&self, value: &MlxTensor) -> Vec<usize> {
        value
            .as_array()
            .shape()
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap_or(0))
            .collect()
    }

    fn gather_rows(&mut self, value: &MlxTensor, rows: &[usize]) -> Result<MlxTensor, Self::Error> {
        if value.as_array().ndim() != 2
            || rows
                .iter()
                .any(|row| i32::try_from(*row).map_or(true, |row| row >= value.as_array().dim(0)))
        {
            return Err(Error::ArchitectureModel(
                "expert route row gather exceeds rank-two input geometry".into(),
            ));
        }
        let rows = self.indices(rows, false)?;
        value
            .as_array()
            .take_axis(&rows, 0, &self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn gather_route_values(
        &mut self,
        value: &MlxTensor,
        flattened_routes: &[usize],
    ) -> Result<MlxTensor, Self::Error> {
        if value.as_array().ndim() != 2
            || flattened_routes
                .iter()
                .any(|position| *position >= value.as_array().size())
        {
            return Err(Error::ArchitectureModel(
                "expert route value gather exceeds rank-two selection geometry".into(),
            ));
        }
        let positions = self.indices(flattened_routes, false)?;
        let rows = i32::try_from(flattened_routes.len()).map_err(|_| {
            Error::ArchitectureModel("expert route count exceeds MLX i32 geometry".into())
        })?;
        value
            .as_array()
            .reshape(&[-1], &self.stream)?
            .take_axis(&positions, 0, &self.stream)?
            .reshape(&[rows, 1], &self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn scatter_add_rows(
        &mut self,
        value: MlxTensor,
        destination_rows: &[usize],
        output_rows: usize,
        reduction: eredu_nn::GroupReduction,
    ) -> Result<MlxTensor, Self::Error> {
        if value.as_array().ndim() != 2
            || usize::try_from(value.as_array().dim(0)).ok() != Some(destination_rows.len())
            || destination_rows.iter().any(|row| *row >= output_rows)
        {
            return Err(Error::ArchitectureModel(
                "expert route scatter-add differs from source-token geometry".into(),
            ));
        }
        let output_rows = i32::try_from(output_rows).map_err(|_| {
            Error::ArchitectureModel("expert route output rows exceed MLX i32 geometry".into())
        })?;
        let mut output = zeros_dtype(
            &[output_rows, value.as_array().dim(1)],
            value.as_array().dtype(),
            &self.stream,
        )?;
        if reduction == eredu_nn::GroupReduction::SequentialGroupOrder {
            // Each wave has at most one contribution per destination. Keeping
            // additions in separate operations preserves rounding after every
            // contribution without serializing unrelated source rows.
            let mut counts = vec![0usize; output_rows as usize];
            let mut waves = Vec::<Vec<usize>>::new();
            for (row, &destination) in destination_rows.iter().enumerate() {
                let wave = counts[destination];
                if wave == waves.len() {
                    waves.push(Vec::new());
                }
                waves[wave].push(row);
                counts[destination] += 1;
            }
            for rows in waves {
                let destinations = rows
                    .iter()
                    .map(|row| destination_rows[*row])
                    .collect::<Vec<_>>();
                let indices = self.indices(&destinations, true)?;
                let selected = self.gather_rows(&value, &rows)?;
                output = output.scatter_add(&indices, selected.as_array(), 0, &self.stream)?;
            }
            return Ok(MlxTensor::from_array(output));
        }
        let indices = self.indices(destination_rows, true)?;
        output
            .scatter_add(&indices, value.as_array(), 0, &self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}
