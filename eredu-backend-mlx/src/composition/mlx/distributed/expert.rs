//! MLX routed-expert mechanisms shared by distributed composition.

use eredu_runtime::ExpertRouteTensorMovement;
use safemlx::{ops::zeros_dtype, Array, Stream};

use crate::{backend::error::Error, MlxTensor};

/// MLX arbitrary-row movement for architecture-owned expert exchange.
#[derive(Debug, Clone)]
pub(crate) struct MlxExpertRouteTensorMovement {
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
        let indices = self.indices(destination_rows, true)?;
        let output = zeros_dtype(
            &[output_rows, value.as_array().dim(1)],
            value.as_array().dtype(),
            &self.stream,
        )?;
        output
            .scatter_add(&indices, value.as_array(), 0, &self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use eredu_nn::Tensor;

    use super::*;

    #[test]
    #[ignore = "requires local MLX native execution"]
    fn mlx_expert_route_movement_preserves_gather_order_and_additive_scatter() {
        let stream = crate::test_stream();
        let input =
            MlxTensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], stream).unwrap();
        let mut movement = MlxExpertRouteTensorMovement::new(stream);

        let gathered = movement.gather_rows(&input, &[2, 0, 2]).unwrap();
        assert_eq!(
            gathered.to_f32_vec(stream).unwrap(),
            [5.0, 6.0, 1.0, 2.0, 5.0, 6.0]
        );

        let scattered = movement.scatter_add_rows(gathered, &[1, 0, 1], 2).unwrap();
        assert_eq!(
            scattered.to_f32_vec(stream).unwrap(),
            [1.0, 2.0, 10.0, 12.0]
        );
    }

    #[test]
    #[ignore = "requires local MLX native execution"]
    fn mlx_expert_route_movement_gathers_flattened_route_values() {
        let stream = crate::test_stream();
        let routes = MlxTensor::from_f32_slice(&[0.1, 0.2, 0.3, 0.4], &[2, 2], stream).unwrap();
        let mut movement = MlxExpertRouteTensorMovement::new(stream);

        let gathered = movement.gather_route_values(&routes, &[3, 0, 2]).unwrap();
        assert_eq!(gathered.as_array().shape(), [3, 1]);
        assert_eq!(gathered.to_f32_vec(stream).unwrap(), [0.4, 0.1, 0.3]);
    }

    #[test]
    #[ignore = "requires local MLX native execution"]
    fn mlx_expert_route_movement_handles_empty_rows_and_rejects_invalid_indices() {
        let stream = crate::test_stream();
        let empty =
            MlxTensor::from_array(zeros_dtype(&[0, 2], safemlx::Dtype::Float32, stream).unwrap());
        let mut movement = MlxExpertRouteTensorMovement::new(stream);

        let gathered = movement.gather_rows(&empty, &[]).unwrap();
        let scattered = movement.scatter_add_rows(gathered, &[], 2).unwrap();
        assert_eq!(scattered.as_array().shape(), [2, 2]);
        assert_eq!(scattered.to_f32_vec(stream).unwrap(), [0.0; 4]);

        let input = MlxTensor::from_f32_slice(&[1.0, 2.0], &[1, 2], stream).unwrap();
        assert!(movement.gather_rows(&input, &[1]).is_err());
        assert!(movement.gather_route_values(&input, &[2]).is_err());
    }
}
