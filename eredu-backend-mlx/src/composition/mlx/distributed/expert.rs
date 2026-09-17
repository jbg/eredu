//! MLX routed-expert mechanisms shared by distributed composition.

use eredu_runtime::{ExpertRouteTensorMovement, PreparedExpertMovementLoan, ExpertRouteMovementSourceError};
use crate::backend::runtime::distributed::topology::original_source::control::OriginalExpertMovementSource;
use safemlx::{ops::zeros_dtype, Array, Stream};

use crate::{backend::error::Error, MlxTensor};
use crate::backend::nn::expert_movement;

/// MLX arbitrary-row movement for architecture-owned expert exchange.
#[derive(Debug, Clone)]
pub(crate) struct MlxExpertRouteTensorMovement {
    stream: Stream,
    source: Option<OriginalExpertMovementSource>,
}

impl MlxExpertRouteTensorMovement {
    /// Binds tensor movement to one caller-owned execution stream.
    pub(crate) fn new(stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            source: None,
        }
    }

    fn destination<T>(&self, count: usize) -> Result<Vec<T>, Error> {
        if let Some(source) = &self.source { return source.destination(count, &self.stream); }
        let mut value = Vec::new();
        value.try_reserve_exact(count).map_err(|cause| Error::ArchitectureModel(cause.to_string()))?;
        Ok(value)
    }
    fn add_rows(&self,base:&MlxTensor,rows:&[usize],updates:&MlxTensor)->Result<MlxTensor,Error> {
        if let Some(source)=&self.source {return source.add(base,rows,updates,&self.stream);}
        let indices=MlxTensor::from_array(self.indices(rows,true)?);
        expert_movement::add(&expert_movement::Native(&self.stream),base,&indices,updates).map_err(Error::Neural)
    }
    fn indices(&self, values: &[usize], trailing_axis: bool) -> Result<Array, Error> {
        if let Some(source) = &self.source { return source.indices(values, trailing_axis, &self.stream); }
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

    fn with_prepared_region<R, E, F>(&mut self, source: Option<PreparedExpertMovementLoan<'_>>, run: F)
        -> Result<Result<R, E>, ExpertRouteMovementSourceError<Error>>
    where F: FnOnce(&mut Self) -> Result<R, E> {
        let source = source.map(|source| OriginalExpertMovementSource::from_loan(source, &self.stream))
            .transpose().map_err(ExpertRouteMovementSourceError::Backend)?;
        if self.source.is_some() {
            return Err(ExpertRouteMovementSourceError::Backend(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)));
        }
        struct Loan<'a> { worker: &'a mut MlxExpertRouteTensorMovement, previous: Option<OriginalExpertMovementSource> }
        impl Drop for Loan<'_> { fn drop(&mut self) { self.worker.source = self.previous.take(); } }
        if let Some(source) = &source {
            source.reserve_call::<(R, E, F, Loan<'_>)>(&self.stream).map_err(ExpertRouteMovementSourceError::Backend)?;
        }
        let previous = std::mem::replace(&mut self.source, source);
        let loan = Loan { worker: self, previous };
        Ok(run(loan.worker))
    }
    fn validate_population(&self,population:eredu_nn::workspace::WorkspaceExpertMovementPopulation,
        transfers:eredu_nn::workspace::WorkspaceExpertTransfers)->Result<(),Error> {
        if let Some(source)=&self.source {source.validate_population(population,transfers,&self.stream)?;}
        Ok(())
    }
    fn index_directory(&self, count: usize) -> Result<Vec<usize>, Error> {
        if let Some(source) = &self.source {
            // Fixed shared order builder and iterative heap-sort controls.
            source.reserve_call::<([usize; 12], [Vec<usize>; 2], &[usize])>(&self.stream)?;
        }
        self.destination(count)
    }
    fn checked_shape(&self, value: &MlxTensor) -> Result<Vec<usize>, Error> {
        let mut shape = self.destination(value.as_array().ndim())?;
        for &dimension in value.as_array().shape() {
            shape.push(usize::try_from(dimension).map_err(|_| Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?);
        }
        Ok(shape)
    }

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
        if let Some(source)=&self.source { return source.gather(value,rows,false,&self.stream); }
        let rows = MlxTensor::from_array(self.indices(rows, false)?);
        expert_movement::gather(&expert_movement::Native(&self.stream), value, &rows, false).map_err(Error::Neural)
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
        if let Some(source)=&self.source { return source.gather(value,flattened_routes,true,&self.stream); }
        let positions = MlxTensor::from_array(self.indices(flattened_routes, false)?);
        expert_movement::gather(&expert_movement::Native(&self.stream), value, &positions, true).map_err(Error::Neural)
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
        let mut output = match &self.source {
            Some(source)=>source.zeros(&value,output_rows,&self.stream)?,
            None=>expert_movement::zeros(&expert_movement::Native(&self.stream), &value, output_rows).map_err(Error::Neural)?,
        };
        if reduction == eredu_nn::GroupReduction::SequentialGroupOrder {
            // Each wave has at most one contribution per destination. Keeping
            // additions in separate operations preserves rounding after every
            // contribution without serializing unrelated source rows.
            let mut counts = self.destination(output_rows as usize)?;
            counts.resize(output_rows as usize, 0usize);
            for &destination in destination_rows { counts[destination] += 1; }
            let wave_count = counts.iter().copied().max().unwrap_or(0);
            let mut wave_lengths = self.destination(wave_count)?;
            wave_lengths.resize(wave_count, 0usize);
            for &count in &counts { for length in &mut wave_lengths[..count] { *length += 1; } }
            let mut waves = self.destination::<Vec<usize>>(wave_count)?;
            for count in wave_lengths { waves.push(self.destination(count)?); }
            counts.fill(0);
            for (row, &destination) in destination_rows.iter().enumerate() {
                let wave = counts[destination];
                waves[wave].push(row);
                counts[destination] += 1;
            }
            for rows in waves {
                let mut destinations = self.destination(rows.len())?;
                destinations.extend(rows.iter().map(|row| destination_rows[*row]));
                let selected = self.gather_rows(&value, &rows)?;
                output = self.add_rows(&output,&destinations,&selected)?;
            }
            return Ok(output);
        }
        self.add_rows(&output,destination_rows,&value)
    }
}

#[cfg(test)]
mod tests {
    use eredu_nn::Tensor;

    use super::*;

    #[test]
    fn sequential_row_reduction_rounds_each_contribution_without_mixing_destinations() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let input = MlxTensor::from_array(
            Array::from_slice(&[256.0f32, 1.0, 1.0, 2.0, -256.0, 3.0], &[6, 1])
                .as_dtype(safemlx::Dtype::Bfloat16, &stream)
                .unwrap(),
        );
        let mut movement = MlxExpertRouteTensorMovement::new(&stream);
        let output = movement
            .scatter_add_rows(
                input,
                &[0, 1, 0, 1, 0, 1],
                3,
                eredu_nn::GroupReduction::SequentialGroupOrder,
            )
            .unwrap();
        assert_eq!(output.to_f32_vec(&stream).unwrap(), [0.0, 6.0, 0.0]);
        let empty =
            MlxTensor::from_array(zeros_dtype(&[0, 1], safemlx::Dtype::Bfloat16, &stream).unwrap());
        let output = movement
            .scatter_add_rows(
                empty,
                &[],
                2,
                eredu_nn::GroupReduction::SequentialGroupOrder,
            )
            .unwrap();
        assert_eq!(output.to_f32_vec(&stream).unwrap(), [0.0, 0.0]);
    }

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

        let scattered = movement
            .scatter_add_rows(gathered, &[1, 0, 1], 2, eredu_nn::GroupReduction::Sum)
            .unwrap();
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
        let scattered = movement
            .scatter_add_rows(gathered, &[], 2, eredu_nn::GroupReduction::Sum)
            .unwrap();
        assert_eq!(scattered.as_array().shape(), [2, 2]);
        assert_eq!(scattered.to_f32_vec(stream).unwrap(), [0.0; 4]);

        let input = MlxTensor::from_f32_slice(&[1.0, 2.0], &[1, 2], stream).unwrap();
        assert!(movement.gather_rows(&input, &[1]).is_err());
        assert!(movement.gather_route_values(&input, &[2]).is_err());
    }
}
